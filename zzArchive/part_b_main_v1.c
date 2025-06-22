// pi2_compare.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zmq.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>

#define FEATURE_DIM 40
#define NUM_SNAPSHOTS 5000
#define K_NEAREST 5

typedef struct {
    int frame_index;
    float features[FEATURE_DIM];
} SnapshotFeature;

// Compute squared Euclidean distance between two feature vectors.
float compute_distance(const SnapshotFeature *a, const SnapshotFeature *b) {
    float dist = 0.0f;
    for (int i = 0; i < FEATURE_DIM; i++) {
        float d = a->features[i] - b->features[i];
        dist += d * d;
    }
    return dist;
}

// Linear search to find k nearest neighbors using OpenMP for parallel distance computation.
void find_nearest_neighbors(const SnapshotFeature *new_feat,
                              const SnapshotFeature *db, int db_size,
                              int k, int *indices_out, float *distances_out) {
    float *dists = malloc(db_size * sizeof(float));
    if (!dists) {
        perror("Allocating distances");
        exit(EXIT_FAILURE);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < db_size; i++) {
        dists[i] = compute_distance(new_feat, &db[i]);
    }
    
    // Initialize the k nearest.
    for (int i = 0; i < k; i++) {
        indices_out[i] = -1;
        distances_out[i] = 1e30f;
    }
    
    // Update k nearest using linear search.
    for (int i = 0; i < db_size; i++) {
        float dist = dists[i];
        int max_index = 0;
        float max_dist = distances_out[0];
        for (int j = 1; j < k; j++) {
            if (distances_out[j] > max_dist) {
                max_dist = distances_out[j];
                max_index = j;
            }
        }
        if (dist < max_dist) {
            distances_out[max_index] = dist;
            indices_out[max_index] = i;
        }
    }
    free(dists);
}

int main(void) {
    // Simulate a precomputed database of 5000 snapshot feature vectors.
    SnapshotFeature *db = malloc(NUM_SNAPSHOTS * sizeof(SnapshotFeature));
    if (!db) {
        perror("Allocating snapshot database");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < NUM_SNAPSHOTS; i++) {
        db[i].frame_index = i;
        for (int j = 0; j < FEATURE_DIM; j++) {
            db[i].features[j] = (float)rand()/(float)RAND_MAX;
        }
    }
    
    // Initialize ZeroMQ context and subscriber.
    void *context = zmq_ctx_new();
    void *subscriber = zmq_socket(context, ZMQ_SUB);
    // Replace <PI1_IP> with the IP address of the first Raspberry Pi.
    if (zmq_connect(subscriber, "tcp://<PI1_IP>:5555") != 0) {
        perror("zmq_connect");
        exit(EXIT_FAILURE);
    }
    zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);
    
    while (1) {
        SnapshotFeature new_feat;
        if (zmq_recv(subscriber, &new_feat, sizeof(new_feat), 0) == -1) {
            perror("zmq_recv");
            continue;
        }
        printf("Received snapshot frame %d\n", new_feat.frame_index);
        
        int indices[K_NEAREST];
        float distances[K_NEAREST];
        find_nearest_neighbors(&new_feat, db, NUM_SNAPSHOTS, K_NEAREST, indices, distances);
        
        printf("Nearest neighbors for frame %d:\n", new_feat.frame_index);
        for (int i = 0; i < K_NEAREST; i++) {
            if (indices[i] != -1)
                printf("  Neighbor: frame %d, squared distance: %f\n", db[indices[i]].frame_index, distances[i]);
        }
    }
    
    zmq_close(subscriber);
    zmq_ctx_destroy(context);
    free(db);
    return 0;
}
