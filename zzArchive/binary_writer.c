#include <stdio.h>
#include <stdlib.h>
#include "binary_writer.h"

void write_group_records(const char *filename, int frame_index, int group_id, int num_pixels, const PixelRecord *records) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open file for binary output");
        return;
    }
    
    /* Write header: frame index, group id, and number of pixels */
    if (fwrite(&frame_index, sizeof(frame_index), 1, fp) != 1 ||
        fwrite(&group_id, sizeof(group_id), 1, fp) != 1 ||
        fwrite(&num_pixels, sizeof(num_pixels), 1, fp) != 1) {
        perror("Failed to write group header");
        fclose(fp);
        return;
    }
    
    /* Write the entire array of PixelRecord */
    if (fwrite(records, sizeof(PixelRecord), num_pixels, fp) != num_pixels) {
        perror("Failed to write pixel records");
        fclose(fp);
        return;
    }
    
    fclose(fp);
}
