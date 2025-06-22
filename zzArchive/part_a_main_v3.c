/*
 * single_daemon.c
 * Monolithic C daemon for Raspberry Pi with pipelined capture, transform, and send
 * using half-precision floats for both visual and audio data to reduce payload.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <signal.h>
 #include <stdatomic.h>
 #include <unistd.h>
 #include <string.h>
 #include <fcntl.h>
 #include <sys/ioctl.h>
 #include <linux/videodev2.h>
 #include <sys/mman.h>
 #include <sys/socket.h>
 #include <arpa/inet.h>
 #include <alsa/asoundlib.h>
 #include <stdint.h>
 #include <pthread.h>
 #include <math.h>
 
 #define VIDEO_DEVICE   "/dev/video0"
 #define AUDIO_DEVICE   "default"
 #define FRAME_WIDTH    640
 #define FRAME_HEIGHT   480
 #define PIXEL_FORMAT   V4L2_PIX_FMT_RGB24
 #define AUDIO_RATE     44100
 #define AUDIO_CHANNELS 1
 #define AUDIO_FRAMES   1024
 #define UDP_PORT       5000
 #define UDP_IP         "192.168.1.2"
 #define QUEUE_SIZE     3
 
 static atomic_bool keep_running = true;
 void handle_sigterm(int signo) { keep_running = false; }
 
 struct buffer { void *start; size_t length; };
 
 // packet holding half-precision visual and audio data
 typedef struct {
     uint16_t *color;    // size FRAME_WIDTH*FRAME_HEIGHT (half-precision)
     uint16_t *audio;    // size AUDIO_FRAMES       (half-precision)
 } FramePacket;
 
 static FramePacket queue[QUEUE_SIZE];
 static int q_head=0, q_tail=0, q_count=0;
 static pthread_mutex_t q_mutex = PTHREAD_MUTEX_INITIALIZER;
 static pthread_cond_t  q_not_empty = PTHREAD_COND_INITIALIZER;
 static pthread_cond_t  q_not_full  = PTHREAD_COND_INITIALIZER;
 
 void enqueue(FramePacket *pkt) {
     pthread_mutex_lock(&q_mutex);
     while (q_count == QUEUE_SIZE && keep_running) {
         pthread_cond_wait(&q_not_full, &q_mutex);
     }
     if (!keep_running) { pthread_mutex_unlock(&q_mutex); return; }
     queue[q_head] = *pkt;
     q_head = (q_head+1) % QUEUE_SIZE;
     q_count++;
     pthread_cond_signal(&q_not_empty);
     pthread_mutex_unlock(&q_mutex);
 }
 
 int dequeue(FramePacket *pkt) {
     pthread_mutex_lock(&q_mutex);
     while (q_count == 0 && keep_running) {
         pthread_cond_wait(&q_not_empty, &q_mutex);
     }
     if (q_count == 0 && !keep_running) { pthread_mutex_unlock(&q_mutex); return 0; }
     *pkt = queue[q_tail];
     q_tail = (q_tail+1) % QUEUE_SIZE;
     q_count--;
     pthread_cond_signal(&q_not_full);
     pthread_mutex_unlock(&q_mutex);
     return 1;
 }
 
 // Convert float32 to IEEE754 half-precision (uint16_t)
 uint16_t float_to_half(float f) {
     uint32_t x = *(uint32_t*)&f;
     uint32_t sign = (x >> 16) & 0x8000;
     int32_t exp = ((x >> 23) & 0xFF) - 127;
     uint32_t mant = x & 0x007FFFFF;
     if (exp > 15) return sign | 0x7C00;
     if (exp > -15) {
         exp += 15;
         mant >>= 13;
         return sign | (exp << 10) | mant;
     }
     return sign;
 }
 
 int init_v4l2(int *fd, struct buffer **buffers, unsigned int *nbufs) {
     struct v4l2_format fmt={0}; struct v4l2_requestbuffers req={0};
     *fd = open(VIDEO_DEVICE, O_RDWR);
     if (*fd<0) return -1;
     fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
     fmt.fmt.pix.width=FRAME_WIDTH; fmt.fmt.pix.height=FRAME_HEIGHT;
     fmt.fmt.pix.pixelformat=PIXEL_FORMAT; fmt.fmt.pix.field=V4L2_FIELD_NONE;
     ioctl(*fd, VIDIOC_S_FMT, &fmt);
     req.count=2; req.type=fmt.type; req.memory=V4L2_MEMORY_MMAP;
     ioctl(*fd, VIDIOC_REQBUFS, &req);
     *buffers = calloc(req.count, sizeof(**buffers));
     for (unsigned i=0; i<req.count; i++){
         struct v4l2_buffer buf={0}; buf.type=req.type; buf.memory=req.memory; buf.index=i;
         ioctl(*fd, VIDIOC_QUERYBUF, &buf);
         (*buffers)[i].length=buf.length;
         (*buffers)[i].start=mmap(NULL,buf.length,PROT_READ|PROT_WRITE,MAP_SHARED,*fd,buf.m.offset);
         ioctl(*fd, VIDIOC_QBUF, &buf);
     }
     enum v4l2_buf_type t=V4L2_BUF_TYPE_VIDEO_CAPTURE; ioctl(*fd,VIDIOC_STREAMON,&t);
     *nbufs=req.count; return 0;
 }
 
 int init_alsa(snd_pcm_t **cap) {
     snd_pcm_hw_params_t *p; int err;
     if ((err=snd_pcm_open(cap,AUDIO_DEVICE,SND_PCM_STREAM_CAPTURE,0))<0) return err;
     snd_pcm_hw_params_malloc(&p);
     snd_pcm_hw_params_any(*cap,p);
     snd_pcm_hw_params_set_access(*cap,p,SND_PCM_ACCESS_RW_INTERLEAVED);
     snd_pcm_hw_params_set_format(*cap,p,SND_PCM_FORMAT_S16_LE);
     snd_pcm_hw_params_set_channels(*cap,p,AUDIO_CHANNELS);
     snd_pcm_hw_params_set_rate(*cap,p,AUDIO_RATE,0);
     snd_pcm_hw_params(*cap,p);
     snd_pcm_hw_params_free(p);
     snd_pcm_prepare(*cap);
     return 0;
 }
 
 void* producer(void *arg) {
     int vfd; struct buffer *buffers; unsigned nbufs;
     snd_pcm_t *cap;
     init_v4l2(&vfd,&buffers,&nbufs);
     init_alsa(&cap);
     const uint32_t max_p = 255U*1000000 + 255U*1000 + 255U;
     uint8_t *rgb_buf = malloc(FRAME_WIDTH*FRAME_HEIGHT*3);
     while (keep_running) {
         FramePacket pkt;
         pkt.color = malloc(FRAME_WIDTH*FRAME_HEIGHT * sizeof(uint16_t));
         pkt.audio = malloc(AUDIO_FRAMES * sizeof(uint16_t));
         // video capture
         struct v4l2_buffer bv={0}; bv.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; bv.memory=V4L2_MEMORY_MMAP;
         ioctl(vfd,VIDIOC_DQBUF,&bv);
         memcpy(rgb_buf,buffers[bv.index].start,bv.bytesused);
         ioctl(vfd,VIDIOC_QBUF,&bv);
         // encode pseudo-color, normalize, half-precision
         for (size_t i=0;i<FRAME_WIDTH*FRAME_HEIGHT;i++){
             uint32_t r=rgb_buf[3*i], g=rgb_buf[3*i+1], b=rgb_buf[3*i+2];
             uint32_t pseudo = r*1000000U + g*1000U + b;
             float norm = (float)pseudo / (float)max_p;
             pkt.color[i] = float_to_half(norm);
         }
         // audio capture
         int16_t tmp[AUDIO_FRAMES];
         snd_pcm_readi(cap,tmp,AUDIO_FRAMES);
         for (size_t i=0;i<AUDIO_FRAMES;i++){
             float v = (tmp[i] + 32768.0f) / 65535.0f;
             pkt.audio[i] = float_to_half(v);
         }
         enqueue(&pkt);
     }
     return NULL;
 }
 
 void* consumer(void *arg) {
     int sock = socket(AF_INET,SOCK_DGRAM,0);
     struct sockaddr_in dest={.sin_family=AF_INET,.sin_port=htons(UDP_PORT)};
     inet_aton(UDP_IP,&dest.sin_addr);
     while (1) {
         FramePacket pkt;
         if (!dequeue(&pkt)) break;
         uint32_t vsz = FRAME_WIDTH*FRAME_HEIGHT * sizeof(uint16_t);
         sendto(sock,&vsz,sizeof(vsz),0,(struct sockaddr*)&dest,sizeof(dest));
         sendto(sock,pkt.color,vsz,0,(struct sockaddr*)&dest,sizeof(dest));
         uint32_t asz = AUDIO_FRAMES * sizeof(uint16_t);
         sendto(sock,&asz,sizeof(asz),0,(struct sockaddr*)&dest,sizeof(dest));
         sendto(sock,pkt.audio,asz,0,(struct sockaddr*)&dest,sizeof(dest));
         free(pkt.color);
         free(pkt.audio);
     }
     close(sock);
     return NULL;
 }
 
 int main() {
     signal(SIGTERM,handle_sigterm);
     pthread_t prod, cons;
     pthread_create(&prod,NULL,producer,NULL);
     pthread_create(&cons,NULL,consumer,NULL);
     pthread_join(prod,NULL);
     pthread_cond_broadcast(&q_not_empty);
     pthread_join(cons,NULL);
     return 0;
 }
 