### pi1_pipeline.c
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <net/if.h>
#include <fcntl.h>
#include <semaphore.h>

#define MTU 9000
#define TCP_IP "192.168.1.2"
#define TCP_PORT 8000
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define AUDIO_RATE 44100
#define PATOMS_PER_SNAPSHOT 1000

// Double-buffer structures
unsigned char video_buffers[2][FRAME_WIDTH * FRAME_HEIGHT * 3];
short audio_buffers[2][4096];
int vid_idx = 0, aud_idx = 0;

// Synchronization
typedef struct {
    unsigned char *video_patoms;
    short *audio_patom;
    int ready;
} snapshot_t;
snapshot_t snapshots[2];
sem_t sem_video, sem_audio, sem_send;

// Placeholder optimized functionsoid process_frame_neon(unsigned char *in, unsigned char *out_patoms) {
    // CC + segmentation via NEON intrinsics
}

void fft_audio(short *in, float *out_patom) {
    // FFT using optimized C FFT library
}

int build_snapshot(int vid_i, int aud_i, unsigned char **out_buf) {
    // Serialize video and audio patoms + timestamp
    // Return size
    *out_buf = malloc(1024);
    return 1024;
}

void *camera_thread(void *arg) {
    while (1) {
        // Capture frame via V4L2 or libcamera into video_buffers[vid_idx]
        sem_post(&sem_video);
        vid_idx ^= 1;
    }
    return NULL;
}

void *audio_thread(void *arg) {
    while (1) {
        // Capture audio via ALSA into audio_buffers[aud_idx]
        sem_post(&sem_audio);
        aud_idx ^= 1;
    }
    return NULL;
}

void *sender_thread(void *arg) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serv;
    serv.sin_family = AF_INET;
    serv.sin_port = htons(TCP_PORT);
    inet_pton(AF_INET, TCP_IP, &serv.sin_addr);
    connect(sock, (struct sockaddr *)&serv, sizeof(serv));
    int val = MTU;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &val, sizeof(val));

    while (1) {
        sem_wait(&sem_video);
        sem_wait(&sem_audio);
        unsigned char *buf;
        int size = build_snapshot(vid_idx ^ 1, aud_idx ^ 1, &buf);
        send(sock, buf, size, 0);
        free(buf);
    }
    close(sock);
    return NULL;
}

int main() {
    pthread_t cam_t, aud_t, snd_t;
    sem_init(&sem_video, 0, 0);
    sem_init(&sem_audio, 0, 0);

    pthread_create(&cam_t, NULL, camera_thread, NULL);
    pthread_create(&aud_t, NULL, audio_thread, NULL);
    pthread_create(&snd_t, NULL, sender_thread, NULL);
    pthread_join(snd_t, NULL);
    return 0;
}
```

### pi2_pipeline.c
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <semaphore.h>
#include "kdtree.h"  // Your kd-tree implementation header

#define TCP_PORT 8000
#define MTU 9000
#define REFs 5000
#define DIM 32
#define QUERIES 1000

float ref_vectors[REFs][DIM];
kd_tree_t *tree;

typedef struct {
    float patom_features[QUERIES][DIM];
} snapshot_t;

snapshot_t *queue[4];
int q_head=0, q_tail=0;
pthread_mutex_t q_mutex;
sem_t q_sem;

float rand_proj_matrix[DIM][DIM];  // Random projection stub

void *worker(void *arg) {
    while (1) {
        sem_wait(&q_sem);
        pthread_mutex_lock(&q_mutex);
        snapshot_t *snap = queue[q_tail++];
        q_tail %= 4;
        pthread_mutex_unlock(&q_mutex);

        float proj_queries[QUERIES][DIM];
        // On-the-fly projection: snap->patom_features * rand_proj_matrix
        for (int i=0;i<QUERIES;i++) {
            for (int d=0;d<DIM;d++) {
                proj_queries[i][d] = 0;
                for (int k=0;k<DIM;k++) {
                    proj_queries[i][d] += snap->patom_features[i][k] * rand_proj_matrix[k][d];
                }
            }
        }
        // KD-tree query
        int indices[QUERIES][5];
        float dists[QUERIES][5];
        for (int i=0;i<QUERIES;i++) {
            kd_query(tree, proj_queries[i], 5, indices[i], dists[i]);
        }
        // Send results to Pi 3
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in serv;
        serv.sin_family = AF_INET;
        serv.sin_port = htons(8001);
        inet_pton(AF_INET, "192.168.1.3", &serv.sin_addr);
        connect(sock, (struct sockaddr *)&serv, sizeof(serv));
        // serialize and send indices/dists
        close(sock);
        free(snap);
    }
    return NULL;
}

void *receiver(void *arg) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(sock, (struct sockaddr *)&addr, sizeof(addr));
    listen(sock,1);
    int conn = accept(sock, NULL, NULL);
    setsockopt(conn, SOL_SOCKET, SO_RCVBUF, &(int){MTU}, sizeof(int));

    while (1) {
        snapshot_t *snap = malloc(sizeof(snapshot_t));
        // recv and parse into snap->patom_features
        pthread_mutex_lock(&q_mutex);
        queue[q_head++] = snap;
        q_head %= 4;
        pthread_mutex_unlock(&q_mutex);
        sem_post(&q_sem);
    }
    return NULL;
}

int main() {
    pthread_t recv_t, workers[4];
    pthread_mutex_init(&q_mutex, NULL);
    sem_init(&q_sem, 0, 0);

    // Load ref_vectors, init rand_proj_matrix, build kd-tree
    tree = kd_build(ref_vectors, REFs, DIM);

    pthread_create(&recv_t, NULL, receiver, NULL);
    for (int i=0; i<4; i++) pthread_create(&workers[i], NULL, worker, NULL);
    pthread_join(recv_t, NULL);
    return 0;
}
```

### pi3_pipeline.c
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <alsa/asoundlib.h>
#include <semaphore.h>
#include "display.h"  // your framebuffer display header

#define TCP_PORT 8001
#define MTU 9000

// Placeholder structures
typedef struct { /* patom structure */ } patom_t;

void inverse_fft_audio(patom_t *patom, short *out_buf) {
    // Inverse FFT implementation
}

void reconstruct_video(patom_t *patoms, unsigned char *out_frame) {
    // Video reconstruction
}

void *receiver(void *arg) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(sock, (struct sockaddr *)&addr, sizeof(addr));
    listen(sock,1);
    int conn = accept(sock, NULL, NULL);
    setsockopt(conn, SOL_SOCKET, SO_RCVBUF, &(int){MTU}, sizeof(int));
    
    while (1) {
        // receive patom matches
        patom_t *video_patoms; patom_t audio_patom;
        // parse data
        unsigned char frame_buf[FRAME_WIDTH * FRAME_HEIGHT * 3];
        short audio_buf[4096];
        inverse_fft_audio(&audio_patom, audio_buf);
        reconstruct_video(video_patoms, frame_buf);
        // Play audio
        pthread_t audio_t;
        snd_pcm_t *pcm;
        snd_pcm_open(&pcm, "default", SND_PCM_STREAM_PLAYBACK, 0);
        snd_pcm_set_params(pcm, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 1, AUDIO_RATE, 1, 500000);
        snd_pcm_writei(pcm, audio_buf, 4096);
        snd_pcm_close(pcm);
        // Display frame
        show_frame(frame_buf);
    }
    return NULL;
}

int main() {
    receiver(NULL);
    return 0;
}
```
