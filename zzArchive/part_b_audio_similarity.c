#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Compare two doubles for qsort/bsearch
int compare_doubles(const void *p, const void *q) {
    double a = *(const double*)p;
    double b = *(const double*)q;
    if (a < b) return -1;
    else if (a > b) return 1;
    else return 0;
}

// Remove duplicates in a sorted array in-place; return new length
int unique_array(double *arr, int n) {
    if (n == 0) return 0;
    int m = 1;
    for (int i = 1; i < n; i++) {
        if (arr[i] != arr[m - 1]) {
            arr[m++] = arr[i];
        }
    }
    return m;
}

// Compute frequency-based match scores (exact and near) between two lists of frequencies
void compute_frequency_scores(const double *f1, int n1,
                              const double *f2, int n2,
                              double freqTol,
                              double *exactScore,
                              double *nearScore) {
    // Copy and sort inputs
    double *a = malloc(n1 * sizeof(double));
    double *b = malloc(n2 * sizeof(double));
    memcpy(a, f1, n1 * sizeof(double));
    memcpy(b, f2, n2 * sizeof(double));
    qsort(a, n1, sizeof(double), compare_doubles);
    qsort(b, n2, sizeof(double), compare_doubles);

    // Unique frequencies
    int u1 = unique_array(a, n1);
    int u2 = unique_array(b, n2);

    // Merge for union of distinct frequencies
    int maxu = u1 + u2;
    double *u = malloc(maxu * sizeof(double));
    int i = 0, j = 0, k = 0;
    while (i < u1 && j < u2) {
        if (a[i] < b[j])      u[k++] = a[i++];
        else if (b[j] < a[i]) u[k++] = b[j++];
        else {                u[k++] = a[i]; i++; j++; }
    }
    while (i < u1) u[k++] = a[i++];
    while (j < u2) u[k++] = b[j++];
    int totDistinct = k;

    int exactMatches = 0, nearMatches = 0;
    // For each unique frequency, check exact and near matches
    for (int idx = 0; idx < totDistinct; idx++) {
        double val = u[idx];
        // Exact if present in both arrays
        if (bsearch(&val, a, u1, sizeof(double), compare_doubles) != NULL &&
            bsearch(&val, b, u2, sizeof(double), compare_doubles) != NULL) {
            exactMatches++;
            nearMatches++;
        } else {
            // Near: binary-search for any in b within ±freqTol
            int low = 0, high = u2 - 1;
            while (low <= high) {
                int mid = (low + high) >> 1;
                if (b[mid] < val - freqTol) low = mid + 1;
                else high = mid - 1;
            }
            if (low < u2 && fabs(b[low] - val) <= freqTol) {
                nearMatches++;
            }
        }
    }

    *exactScore = (double)exactMatches / totDistinct;
    *nearScore  = (double)nearMatches  / totDistinct;

    free(a);
    free(b);
    free(u);
}

// Compute amplitude-based match scores (exact and near) over aligned time-series
void compute_amplitude_scores(const double *x, const double *y,
                              int T, double ampTol,
                              double *exactScore,
                              double *nearScore) {
    int exact = 0, near = 0;
    for (int t = 0; t < T; t++) {
        if (x[t] == y[t]) exact++;
        if (fabs(x[t] - y[t]) <= ampTol) near++;
    }
    *exactScore = (double)exact / T;
    *nearScore  = (double)near  / T;
}

int main() {
    // Tolerance thresholds (example values)
    double freqTol = 0.5;  // ±0.5 Hz
    double ampTol  = 0.01; // ±0.01 amplitude

    // Example frequency data
    double f1[] = {100.0, 200.0, 300.5, 400.0};
    double f2[] = {99.8, 200.0, 305.0, 500.0};
    int n1 = sizeof(f1) / sizeof(f1[0]);
    int n2 = sizeof(f2) / sizeof(f2[0]);
    double freqExact, freqNear;
    compute_frequency_scores(f1, n1, f2, n2, freqTol, &freqExact, &freqNear);
    printf("Frequency exact match score: %.4f\n", freqExact);
    printf("Frequency near match score:  %.4f\n", freqNear);

    // Example amplitude data
    double x[] = {0.2, 0.5, 0.7, 0.1};
    double y[] = {0.2, 0.51, 0.68, 0.0};
    int T = sizeof(x) / sizeof(x[0]);
    double ampExact, ampNear;
    compute_amplitude_scores(x, y, T, ampTol, &ampExact, &ampNear);
    printf("Amplitude exact match score: %.4f\n", ampExact);
    printf("Amplitude near match score:  %.4f\n", ampNear);

    return 0;
}
