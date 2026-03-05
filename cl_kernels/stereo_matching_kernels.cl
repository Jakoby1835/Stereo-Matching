__kernel void ncc_stereo_matching(
    __global const unsigned char* left_image,
    __global const unsigned char* right_image,
    __global char* result_image,
    int image_width,
    int image_height,
    int window_half_size,
    int max_disp)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Randpixel überspringen (wie im Original)
    if (x < window_half_size || x >= image_width - window_half_size ||
        y < window_half_size || y >= image_height - window_half_size) {
        return;
    }

    int best_disparity = 0;
    float best_ncc = -1.0f;

    // Suchbereichsgrenzen (exakte Übernahme der Original-Logik)
    int left_limit = window_half_size;
    int right_limit = window_half_size;
    if (x > max_disp) {
        left_limit = max(window_half_size, x - max_disp);
        right_limit = min(image_width - 1 - window_half_size, x + max_disp);
    }

    // Iteration über alle möglichen Disparitäten (d.h. über Zielpixel im rechten Bild)
    for (int e = left_limit; e <= right_limit; ++e) {
        float sum_ref = 0.0f;
        float sum_search = 0.0f;
        float sum_ab = 0.0f;
        float sum_aa = 0.0f;
        float sum_bb = 0.0f;
        int counter = 0;

        // Summiere über das Fenster der Größe (2*window_half_size+1)^2
        for (int wy = -window_half_size; wy <= window_half_size; ++wy) {
            for (int wx = -window_half_size; wx <= window_half_size; ++wx) {
                int ref_idx = (y + wy) * image_width + (x + wx);
                int search_idx = (y + wy) * image_width + (e + wx);

                uchar ref_val = left_image[ref_idx];
                uchar search_val = right_image[search_idx];

                sum_ref += ref_val;
                sum_search += search_val;
                sum_ab += ref_val * search_val;
                sum_aa += ref_val * ref_val;
                sum_bb += search_val * search_val;
                ++counter;
            }
        }

        // Mittelwerte und Varianzen für NCC
        float mean_ref = sum_ref / counter;
        float mean_search = sum_search / counter;
        float mean_ab = sum_ab / counter;
        float mean_aa = sum_aa / counter;
        float mean_bb = sum_bb / counter;

        // Normalized Cross-Correlation (NCC)
        float numerator = mean_ab - mean_ref * mean_search;
        float denom_a = mean_aa - mean_ref * mean_ref;
        float denom_b = mean_bb - mean_search * mean_search;
        float denom = sqrt(denom_a * denom_b);  // Achtung: Wie im Original ohne Schutz gegen negative Werte
        float ncc = numerator / denom;

        // Höhere NCC-Werte bedeuten bessere Übereinstimmung
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_disparity = x - e;  // Disparität = Differenz der x-Positionen
        }
    }

    // Ergebnis schreiben
    int pixel_offset = y * image_width + x;
    result_image[pixel_offset] = (char)best_disparity;
}