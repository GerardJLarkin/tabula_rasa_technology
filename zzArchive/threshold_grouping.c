/*
 * single_daemon.c
 * Monolithic C daemon on Raspberry Pi: capture video/audio, transform, group, and stream.
 * - Visual: pseudo-color (R,G,B → 9-digit), normalize, half-float, threshold grouping.
 * - Audio: FFT magnitudes, normalize, half-float.
 * - Adds sequence ID 0–899 per snapshot.
 */

 #define _GNU_SOURCE
 #include <stdio.h>
 #include <stdlib.h>
 #include <stdbool.h>
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
 #include <complex.h>
 
 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif
 
 #define VIDEO_DEVICE   "/dev/video0"
 #define AUDIO_DEVICE   "default"
 #define FRAME_WIDTH    640
 #define FRAME_HEIGHT   480
 #define PIXEL_FORMAT   V4L2_PIX_FMT_RGB24
 #define AUDIO_RATE     44100
 #define AUDIO_CHANNELS 1
 #define AUDIO_FRAMES   1024
 #define FFT_SIZE       AUDIO_FRAMES
 #define UDP_PORT       5000
 #define UDP_IP         "192.168.0.172"
 #define QUEUE_SIZE     3
 #define THRESHOLD      0.0005f
 
 struct buffer { void *start; size_t length; };
 
 static atomic_bool keep_running = ATOMIC_VAR_INIT(true);
 static uint16_t seq_counter = 0;
 
 void handle_sigint(int signo) {
     (void)signo;
     keep_running = false;
     system("shutdown -h now");
 }
 void handle_sigterm(int signo) {
     (void)signo;
     keep_running = false;
 }
 
 // Forward declarations
 int threshold_group(const float *data, int width, int height, float threshold, int *labels);
 uint16_t float_to_half(float f);
 void fft(float complex *x, int n); // void measn function does not return a value after execution
 
 // Disjoint-set globals
 static int *dsu_parent; // static = can't be accessed outside this file, * = means variable is a pointer to a memory location not the variable
 static int *dsu_rank;
 
 static inline int dsu_find(int x) { // inline improves performance?
     while (dsu_parent[x] != x) {
         dsu_parent[x] = dsu_parent[dsu_parent[x]];
         x = dsu_parent[x];
     }
     return x;
 }
 static inline void dsu_union(int a, int b) {
     a = dsu_find(a);
     b = dsu_find(b);
     if (a == b) return;
     if (dsu_rank[a] < dsu_rank[b]) dsu_parent[a] = b;
     else if (dsu_rank[a] > dsu_rank[b]) dsu_parent[b] = a;
     else { dsu_parent[b] = a; dsu_rank[a]++; }
 }
 
 int threshold_group(const float *data, int width, int height, float threshold, int *labels) {
     int N = width * height;
     dsu_parent = malloc(N * sizeof(int));
     dsu_rank   = calloc(N, sizeof(int));
     if (!dsu_parent || !dsu_rank) return 0;
     for (int i = 0; i < N; i++) dsu_parent[i] = i;
     for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++) {
             int idx = y*width + x;
             float v = data[idx];
             if (x+1 < width) {
                 int r = idx+1;
                 if (fabsf(v - data[r]) <= threshold) dsu_union(idx, r);
             }
             if (y+1 < height) {
                 int d = idx+width;
                 if (fabsf(v - data[d]) <= threshold) dsu_union(idx, d);
             }
         }
     }
     int *map = malloc(N * sizeof(int));
     int count = 0;
     for (int i = 0; i < N; i++) map[i] = -1;
     for (int i = 0; i < N; i++) {
         int root = dsu_find(i);
         if (map[root] < 0) map[root] = count++;
         labels[i] = map[root];
     }
     free(dsu_parent);
     free(dsu_rank);
     free(map);
     return count;
 }
 
 void fft(float complex *x, int n) {
     for (int i = 1, j = 0; i < n; i++) {
         int bit = n >> 1;
         for (; j & bit; bit >>= 1) j ^= bit;
         j |= bit;
         if (i < j) { float complex t = x[i]; x[i] = x[j]; x[j] = t; }
     }
     for (int len = 2; len <= n; len <<= 1) {
         float angle = -2.0f * M_PI / len;
         float complex wlen = cexpf(I * angle);
         for (int i = 0; i < n; i += len) {
             float complex w = 1.0f;
             for (int j = 0; j < len/2; j++) {
                 float complex u = x[i+j];
                 float complex v = w * x[i+j+len/2];
                 x[i+j]         = u + v;
                 x[i+j+len/2] = u - v;
                 w *= wlen;
             }
         }
     }
 }
 
 uint16_t float_to_half(float f) {
     uint32_t x = *(uint32_t*)&f;
     uint16_t s = (x >> 16) & 0x8000;
     int32_t e = ((x >> 23) & 0xFF) - 127;
     uint32_t m = x & 0x007FFFFF;
     if (e > 15) return s | 0x7C00;
     if (e > -15) {
         e += 15;
         m >>= 13;
         return s | (e << 10) | m;
     }
     return s;
 }
 
 typedef struct { uint16_t seq_id; uint16_t *color; uint16_t *audio; } Packet;
 static Packet queue[QUEUE_SIZE];
 static int qh=0, qt=0, qc=0;
 static pthread_mutex_t qm = PTHREAD_MUTEX_INITIALIZER;
 static pthread_cond_t  qn = PTHREAD_COND_INITIALIZER;
 static pthread_cond_t  qf = PTHREAD_COND_INITIALIZER;
 
 void enqueue_pkt(Packet *p) {
     pthread_mutex_lock(&qm);
     while (qc == QUEUE_SIZE && keep_running) pthread_cond_wait(&qf, &qm);
     if (!keep_running) { pthread_mutex_unlock(&qm); return; }
     queue[qh] = *p;
     qh = (qh + 1) % QUEUE_SIZE;
     qc++;
     pthread_cond_signal(&qn);
     pthread_mutex_unlock(&qm);
 }
 
 int dequeue_pkt(Packet *p) {
     pthread_mutex_lock(&qm);
     while (qc == 0 && keep_running) pthread_cond_wait(&qn, &qm);
     if (qc == 0 && !keep_running) { pthread_mutex_unlock(&qm); return 0; }
     *p = queue[qt];
     qt = (qt + 1) % QUEUE_SIZE;
     qc--;
     pthread_cond_signal(&qf);
     pthread_mutex_unlock(&qm);
     return 1;
 }
 
 int init_v4l2(int *fd, struct buffer **bufs, unsigned *n) {
     struct v4l2_format fmt = {0};
     struct v4l2_requestbuffers req = {0};
     *fd = open(VIDEO_DEVICE, O_RDWR);
     if (*fd < 0) return -1;
     fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
     fmt.fmt.pix.width = FRAME_WIDTH;
     fmt.fmt.pix.height = FRAME_HEIGHT;
     fmt.fmt.pix.pixelformat = PIXEL_FORMAT;
     fmt.fmt.pix.field = V4L2_FIELD_NONE;
     ioctl(*fd, VIDIOC_S_FMT, &fmt);
     req.count = 2;
     req.type = fmt.type;
     req.memory = V4L2_MEMORY_MMAP;
     ioctl(*fd, VIDIOC_REQBUFS, &req);
     *bufs = calloc(req.count, sizeof(**bufs));
     for (unsigned i = 0; i < req.count; i++) {
         struct v4l2_buffer buf = {0};
         buf.type = req.type;
         buf.memory = req.memory;
         buf.index = i;
         ioctl(*fd, VIDIOC_QUERYBUF, &buf);
         (*bufs)[i].length = buf.length;
         (*bufs)[i].start = mmap(NULL, buf.length, PROT_READ|PROT_WRITE,
                                  MAP_SHARED, *fd, buf.m.offset);
         ioctl(*fd, VIDIOC_QBUF, &buf);
     }
     enum v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
     ioctl(*fd, VIDIOC_STREAMON, &t);
     *n = req.count;
     return 0;
 }
 
 int init_alsa(snd_pcm_t **capture) {
     snd_pcm_hw_params_t *params;
     int err = snd_pcm_open(capture, AUDIO_DEVICE,
                            SND_PCM_STREAM_CAPTURE, 0);
     if (err < 0) return err;
     snd_pcm_hw_params_malloc(&params);
     snd_pcm_hw_params_any(*capture, params);
     snd_pcm_hw_params_set_access(*capture, params,
                                   SND_PCM_ACCESS_RW_INTERLEAVED);
     snd_pcm_hw_params_set_format(*capture, params,
                                  SND_PCM_FORMAT_S16_LE);
     snd_pcm_hw_params_set_channels(*capture, params,
                                    AUDIO_CHANNELS);
     snd_pcm_hw_params_set_rate(*capture, params,
                                AUDIO_RATE, 0);
     snd_pcm_hw_params(*capture, params);
     snd_pcm_hw_params_free(params);
     snd_pcm_prepare(*capture);
     return 0;
 }
 
 void* producer(void *_) {
     int vfd;
     struct buffer *bufs;
     unsigned n;
     snd_pcm_t *cap;
     if (init_v4l2(&vfd, &bufs, &n) < 0) return NULL;
     if (init_alsa(&cap) < 0) return NULL;
     float *norm = malloc(FRAME_WIDTH * FRAME_HEIGHT * sizeof(float));
     float complex spec[FFT_SIZE];
     while (keep_running) {
         Packet pck;
         pck.seq_id = seq_counter;
         if (++seq_counter >= 900) seq_counter = 0;
         struct v4l2_buffer b = {0};
         b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
         b.memory = V4L2_MEMORY_MMAP;
         ioctl(vfd, VIDIOC_DQBUF, &b);
         memcpy(norm, bufs[b.index].start, b.bytesused);
         ioctl(vfd, VIDIOC_QBUF, &b);
         for (int i = 0; i < FRAME_WIDTH * FRAME_HEIGHT; i++) {
             uint8_t *rgb = (uint8_t*)bufs[b.index].start + 3*i;
             uint32_t r = rgb[0], g = rgb[1], bl = rgb[2];
             uint32_t pseudo = r*1000000U + bl*1000U + g;
             norm[i] = (float)pseudo / 255255255.0f;
         }
         int *labels = malloc(FRAME_WIDTH * FRAME_HEIGHT * sizeof(int));
         threshold_group(norm, FRAME_WIDTH, FRAME_HEIGHT,
                         THRESHOLD, labels);
         printf("Snapshot %u grouping (threshold=%.4f):\n", pck.seq_id, THRESHOLD);
         for (int y = 0; y < FRAME_HEIGHT; y++) {
             for (int x = 0; x < FRAME_WIDTH; x++) {
                 int idx = y * FRAME_WIDTH + x;
                 printf("(%d,%d): %.6f label=%d\n", x, y,
                        norm[idx], labels[idx]);
             }
         }
         free(labels);
         pck.color = malloc(FRAME_WIDTH * FRAME_HEIGHT * sizeof(uint16_t));
         for (int i = 0; i < FRAME_WIDTH*FRAME_HEIGHT; i++)
             pck.color[i] = float_to_half(norm[i]);
         int16_t tmp[AUDIO_FRAMES];
         snd_pcm_readi(cap, tmp, AUDIO_FRAMES);
         for (int i = 0; i < FFT_SIZE; i++)
             spec[i] = tmp[i]/32768.0f + 0.0f*I;
         fft(spec, FFT_SIZE);
         printf("Snapshot %u FFT magnitudes:\n", pck.seq_id);
         for (int i = 0; i < FFT_SIZE; i++) {
             float mag = cabsf(spec[i]) / (float)FFT_SIZE;
             printf("Bin %d: %.6f\n", i, mag);
         }
         pck.audio = malloc(FFT_SIZE * sizeof(uint16_t));
         for (int i = 0; i < FFT_SIZE; i++) {
             float mag = cabsf(spec[i]) / (float)FFT_SIZE;
             pck.audio[i] = float_to_half(mag);
         }
         enqueue_pkt(&pck);
     }
     free(norm);
     return NULL;
 }
 
 void* consumer(void *_) {
     int s = socket(AF_INET, SOCK_DGRAM, 0);
     struct sockaddr_in d = {.sin_family = AF_INET,
                              .sin_port = htons(UDP_PORT)};
     inet_aton(UDP_IP, &d.sin_addr);
     while (keep_running || qc > 0) {
         Packet p;
         if (!dequeue_pkt(&p)) continue;
         if (p.seq_id % 30 == 0) {
             printf("[Test] Snapshot %u captured\n", p.seq_id);
             fflush(stdout);
         }
         uint32_t vsz = FRAME_WIDTH * FRAME_HEIGHT * sizeof(uint16_t);
         sendto(s, &p.seq_id, sizeof(p.seq_id), 0,
                (struct sockaddr*)&d, sizeof(d));
         sendto(s, &vsz, sizeof(vsz), 0,
                (struct sockaddr*)&d, sizeof(d));
         sendto(s, p.color, vsz, 0,
                (struct sockaddr*)&d, sizeof(d));
         uint32_t asz = FFT_SIZE * sizeof(uint16_t);
         sendto(s, &asz, sizeof(asz), 0,
                (struct sockaddr*)&d, sizeof(d));
         sendto(s, p.audio, asz, 0,
                (struct sockaddr*)&d, sizeof(d));
         free(p.color);
         free(p.audio);
     }
     close(s);
     return NULL;
 }
 
 int main() {
     signal(SIGTERM, handle_sigterm);
     signal(SIGINT, handle_sigint);
     pthread_t prod, cons;
     pthread_create(&prod, NULL, producer, NULL);
     pthread_create(&cons, NULL, consumer, NULL);
     pthread_join(prod, NULL);
     pthread_cond_broadcast(&qn);
     pthread_join(cons, NULL);
     return 0;
 }
 