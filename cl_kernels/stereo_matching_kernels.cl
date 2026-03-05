kernel void ncc_stereo_matching(
    global const unsigned char* in_image_left,
    global const unsigned char* in_image_right,
    global char* out_disparity_left_to_right,
    int image_width,
    int image_height,
    int window_half_size,
    int max_disp
) {
    // Globale Thread‑ID = Pixelkoordinate
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Randbereiche ignorieren – Fenster muss vollständig im Bild liegen
    if (x < window_half_size || x >= image_width - window_half_size ||
        y < window_half_size || y >= image_height - window_half_size) {
        return;
    }

    int best_disparity = 0;
    float best_ncc = -1.0f;

    // Suchbereich im rechten Bild (Epipolarlinie)
    int left_limit  = max(window_half_size, x - max_disp);
    int right_limit = min(image_width - 1 - window_half_size, x + max_disp);

    // Über alle möglichen Disparitäten (bzw. Zielpositionen e) iterieren
    for (int e = left_limit; e <= right_limit; ++e) {
        float ref_sum = 0.0f;
        float search_sum = 0.0f;
        float var_ab = 0.0f;
        float var_a  = 0.0f;
        float var_b  = 0.0f;
        int counter = 0;

        // Fenster der Grösse (2*window_half_size+1)^2 durchlaufen
        for (int win_y = -window_half_size; win_y <= window_half_size; ++win_y) {
            for (int win_x = -window_half_size; win_x <= window_half_size; ++win_x) {
                int ref_idx  = (y + win_y) * image_width + (x + win_x);
                int search_idx = (y + win_y) * image_width + (e + win_x);

                float ref_val  = in_image_left[ref_idx];
                float search_val = in_image_right[search_idx];

                ref_sum   += ref_val;
                search_sum += search_val;
                var_ab    += ref_val * search_val;
                var_a     += ref_val * ref_val;
                var_b     += search_val * search_val;
                ++counter;
            }
        }

        // Mittelwerte
        float ref_mean  = ref_sum / counter;
        float search_mean = search_sum / counter;

        // Varianzen (mit Korrektur durch Mittelwerte)
        float cov = var_ab / counter - ref_mean * search_mean;
        float ref_var  = var_a / counter - ref_mean * ref_mean;
        float search_var = var_b / counter - search_mean * search_mean;

        // Normalized Cross‑Correlation (NCC)
        // Schutz vor Division durch Null (falls Varianzen nahe 0)
        float ncc = 0.0f;
        float denom = sqrt(ref_var * search_var);
        if (denom > 1e-6f) {
            ncc = cov / denom;
        }

        // Höhere NCC‑Werte bedeuten bessere Übereinstimmung
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_disparity = x - e;   // Disparität = Differenz der x‑Positionen
        }
    }

    // Ergebnis ins Ausgabebild schreiben
    int pixel_offset = y * image_width + x;
    out_disparity_left_to_right[pixel_offset] = (char)best_disparity;
}