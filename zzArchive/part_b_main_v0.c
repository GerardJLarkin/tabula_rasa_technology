#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUM_SNAPSHOTS 5000
#define FEATURE_DIM 40   // Dimension of the precomputed feature vector
#define K_NEAREST 5      // Number of nearest neighbors to retrieve

// Structure for a snapshot's precomputed feature vector.
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

// Find the k nearest neighbors using a linear search, parallelized with OpenMP.
void find_nearest_neighbors(const SnapshotFeature *new_feat,
                              const SnapshotFeature *db, int db_size,
                              int k, int *indices_out, float *distances_out) {
    // Allocate a temporary distance array.
    float *dists = malloc(db_size * sizeof(float));
    if (!dists) {
        perror("Allocating distances");
        exit(EXIT_FAILURE);
    }

    // Compute distances in parallel.
    #pragma omp parallel for
    for (int i = 0; i < db_size; i++) {
        dists[i] = compute_distance(new_feat, &db[i]);
    }
    
    // Initialize the k nearest with large values.
    for (int i = 0; i < k; i++) {
        indices_out[i] = -1;
        distances_out[i] = 1e30f;
    }
    
    // A simple linear search to update the k nearest neighbors.
    for (int i = 0; i < db_size; i++) {
        float dist = dists[i];
        // Find the current maximum in the k-nearest list.
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
    // Simulate a database of 5000 snapshot feature vectors.
    SnapshotFeature *db = malloc(NUM_SNAPSHOTS * sizeof(SnapshotFeature));
    if (!db) {
        perror("Allocating snapshot database");
        exit(EXIT_FAILURE);
    }
    // Fill with dummy data (in your application, these come from Part A's processing).
    for (int i = 0; i < NUM_SNAPSHOTS; i++) {
        db[i].frame_index = i;
        for (int j = 0; j < FEATURE_DIM; j++) {
            // For example, random values in [0, 1].
            db[i].features[j] = (float)rand() / (float)RAND_MAX;
        }
    }
    
    // Simulate a new snapshot's feature vector.
    SnapshotFeature new_feat;
    new_feat.frame_index = NUM_SNAPSHOTS;  // New frame index.
    for (int j = 0; j < FEATURE_DIM; j++) {
        new_feat.features[j] = (float)rand() / (float)RAND_MAX;
    }
    
    // Find the K nearest neighbors.
    int indices[K_NEAREST];
    float distances[K_NEAREST];
    find_nearest_neighbors(&new_feat, db, NUM_SNAPSHOTS, K_NEAREST, indices, distances);
    
    printf("Nearest neighbors for new snapshot (frame %d):\n", new_feat.frame_index);
    for (int i = 0; i < K_NEAREST; i++) {
        if (indices[i] != -1)
            printf("  Neighbor %d: frame %d with squared distance %f\n", i, db[indices[i]].frame_index, distances[i]);
    }
    
    free(db);
    return 0;
}
