/*
 * partC_grouping.c
 *
 * This program runs on a third Raspberry Pi.
 * It receives SnapshotFeature messages (blobs) published by Part A (via ZeroMQ),
 * and groups them into clusters based on similarity.
 * Each group is represented by an archetype feature vector (a running average).
 * Blobs whose squared Euclidean distance from a group's archetype is below a set threshold
 * are added to that group; otherwise, a new group is created.
 *
 * Compile with:
 *   gcc -O3 -march=native -fopenmp -o partC_grouping partC_grouping.c -lzmq -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zmq.h>
#include <math.h>
#include <omp.h>

#define FEATURE_DIM 40
#define GROUP_THRESHOLD 0.05f  // Set threshold (tune as needed)
#define MAX_GROUPS 5000

// Snapshot feature structure received from Part A.
typedef struct {
    int frame_index;
    float features[FEATURE_DIM];
} SnapshotFeature;

// Group structure: each group holds an archetype and count.
typedef struct {
    int group_id;
    int count;
    float archetype[FEATURE_DIM]; // running average of features
} Group;

// Global group array and counter.
Group groups[MAX_GROUPS];
int num_groups = 0;
int next_group_id = 0;

// Compute squared Euclidean distance between two feature vectors.
float compute_distance(const SnapshotFeature *a, const float *b) {
    float dist = 0.0f;
    for (int i = 0; i < FEATURE_DIM; i++) {
        float d = a->features[i] - b[i];
        dist += d * d;
    }
    return dist;
}

// Update group's archetype with a new snapshot feature vector.
void update_group(Group *grp, const SnapshotFeature *new_feat) {
    // New archetype = (old_archetype * count + new_feat) / (count+1)
    for (int i = 0; i < FEATURE_DIM; i++) {
        grp->archetype[i] = (grp->archetype[i] * grp->count + new_feat->features[i]) / (grp->count + 1);
    }
    grp->count++;
}

// Given a new snapshot feature vector, find a matching group.
// Returns the index of the group if found (distance < threshold), or -1 if no match.
int find_matching_group(const SnapshotFeature *new_feat) {
    int match_index = -1;
    float best_dist = 1e30f;
    for (int i = 0; i < num_groups; i++) {
        float dist = compute_distance(new_feat, groups[i].archetype);
        if (dist < GROUP_THRESHOLD && dist < best_dist) {
            best_dist = dist;
            match_index = i;
        }
    }
    return match_index;
}

// Process a new snapshot: group it or create a new group.
void process_snapshot(const SnapshotFeature *new_feat) {
    int idx = find_matching_group(new_feat);
    if (idx >= 0) {
        // Found a matching group: update the archetype.
        update_group(&groups[idx], new_feat);
        printf("Frame %d added to group %d (new count: %d, dist: %.6f)\n",
               new_feat->frame_index, groups[idx].group_id, groups[idx].count, compute_distance(new_feat, groups[idx].archetype));
    } else {
        // No matching group: create a new group.
        if (num_groups >= MAX_GROUPS) {
            fprintf(stderr, "Maximum number of groups reached!\n");
            return;
        }
        groups[num_groups].group_id = next_group_id++;
        groups[num_groups].count = 1;
        memcpy(groups[num_groups].archetype, new_feat->features, FEATURE_DIM * sizeof(float));
        printf("Frame %d created new group %d\n", new_feat->frame_index, groups[num_groups].group_id);
        num_groups++;
    }
}

// Main function: receive snapshots from Part A via ZeroMQ and group them.
int main(void) {
    // Initialize ZeroMQ context and subscriber.
    void *context = zmq_ctx_new();
    void *subscriber = zmq_socket(context, ZMQ_SUB);
    // Connect to Part A's publisher (replace <PI1_IP> with actual IP address).
    if (zmq_connect(subscriber, "tcp://<PI1_IP>:5555") != 0) {
        perror("zmq_connect");
        exit(EXIT_FAILURE);
    }
    // Subscribe to all messages.
    zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);
    
    SnapshotFeature new_feat;
    while (1) {
        // Receive a SnapshotFeature (blocking).
        int bytes = zmq_recv(subscriber, &new_feat, sizeof(new_feat), 0);
        if (bytes == -1) {
            perror("zmq_recv");
            continue;
        }
        printf("Received snapshot frame %d\n", new_feat.frame_index);
        process_snapshot(&new_feat);
    }
    
    zmq_close(subscriber);
    zmq_ctx_destroy(context);
    return 0;
}
