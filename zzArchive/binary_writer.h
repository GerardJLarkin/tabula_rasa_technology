#ifndef BINARY_WRITER_H
#define BINARY_WRITER_H

typedef struct {
    float norm_x;         // Normalized x coordinate [-1, 1]
    float norm_y;         // Normalized y coordinate [-1, 1]
    unsigned int color;   // Pixel color (0xRRGGBB)
    unsigned char neighbor_mask; // 8-bit neighbor mask
    int segment;          // Segment index [0-7]
    int seg_count;        // Total number of pixels in this group that fall into this segment
} PixelRecord;

/* Write an array of PixelRecord to a binary file.
 * The file header includes:
 *    - frame index (int)
 *    - group id (int)
 *    - number of pixel records (int)
 * Then writes the PixelRecord array.
 */
void write_group_records(const char *filename, int frame_index, int group_id, int num_pixels, const PixelRecord *records);

#endif // BINARY_WRITER_H
