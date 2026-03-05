#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <string>

// BMP-Header sind packed, daher keine Padding-Einfügungen
#pragma pack(push, 1)
struct BitmapFileHeader {
    uint16_t bfType;      // "BM"
    uint32_t bfSize;      // Gesamtgröße der Datei
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;   // Offset zu den Pixeldaten
};

struct BitmapInfoHeader {
    uint32_t biSize;           // Größe dieses Headers (40)
    int32_t  biWidth;          // Breite
    int32_t  biHeight;         // Höhe (positiv = bottom-up, negativ = top-down)
    uint16_t biPlanes;         // muss 1 sein
    uint16_t biBitCount;       // Bits pro Pixel (24)
    uint32_t biCompression;    // 0 = BI_RGB (unkomprimiert)
    uint32_t biSizeImage;      // Größe der Pixeldaten (kann 0 sein bei BI_RGB)
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

struct RGB {
    uint8_t r, g, b;
    bool operator==(const RGB& other) const {
        return r == other.r && g == other.g && b == other.b;
    }
};

// Liest eine BMP-Datei und gibt die Pixel als Vektor (zeilenweise von oben nach unten) zurück.
// Bei Fehler wird eine Exception geworfen.
std::vector<RGB> readBMP(const std::string& filename, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Datei kann nicht geöffnet werden: " + filename);
    }

    BitmapFileHeader fileHeader;
    file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    if (fileHeader.bfType != 0x4D42) { // "BM" in little-endian
        throw std::runtime_error("Keine gültige BMP-Datei (falsche Signatur): " + filename);
    }

    BitmapInfoHeader infoHeader;
    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));
    if (infoHeader.biSize != 40) {
        throw std::runtime_error("Nur BMP mit 40-Byte InfoHeader unterstützt: " + filename);
    }
    if (infoHeader.biBitCount != 24) {
        throw std::runtime_error("Nur 24-Bit BMP unterstützt: " + filename);
    }
    if (infoHeader.biCompression != 0) {
        throw std::runtime_error("Nur unkomprimierte BMP (BI_RGB) unterstützt: " + filename);
    }

    width = infoHeader.biWidth;
    height = std::abs(infoHeader.biHeight); // Höhe immer positiv (wir speichern von oben)
    bool bottomUp = (infoHeader.biHeight > 0); // positiv = bottom-up

    int bytesPerPixel = 3;
    int lineSize = (width * bytesPerPixel + 3) & ~3; // Padding auf 4 Byte

    std::vector<RGB> pixels(width * height);

    // Temporärer Puffer für eine Zeile inkl. Padding
    std::vector<uint8_t> row(lineSize);

    for (int y = 0; y < height; ++y) {
        // Datei-Y: bei bottom-up ist Zeile y (von oben) die unterste im File, also height-1-y
        int fileY = bottomUp ? (height - 1 - y) : y;
        file.seekg(fileHeader.bfOffBits + fileY * lineSize, std::ios::beg);

        file.read(reinterpret_cast<char*>(row.data()), lineSize);
        if (!file) {
            throw std::runtime_error("Fehler beim Lesen der Pixeldaten: " + filename);
        }

        for (int x = 0; x < width; ++x) {
            // BMP speichert in BGR-Reihenfolge
            uint8_t b = row[x * 3 + 0];
            uint8_t g = row[x * 3 + 1];
            uint8_t r = row[x * 3 + 2];
            pixels[y * width + x] = { r, g, b };
        }
    }

    return pixels;
}

int main(int argc, char* argv[]) {

    /////////////////////////////////////////////////////////

    //Settings
    std::string file1 = "..\\BMPCompare\\Images_to_compare\\seq_quarter.bmp";
    
    std::string file2 = "..\\BMPCompare\\Images_to_compare\\opencl_quarter.bmp";

    int ignore_border = 160; //zu ingorierende Randpixel

    ////////////////////////////////////////////

    std::cout << "Bild 1: " << file1 << "\n Bild 2: " << file2 << "\n Zu ignorierende Randpixel: " << ignore_border << std::endl;

    try {
        int w1, h1, w2, h2;
        std::vector<RGB> img1 = readBMP(file1, w1, h1);
        std::vector<RGB> img2 = readBMP(file2, w2, h2);

        if (w1 != w2 || h1 != h2) {
            std::cerr << "Bilder haben unterschiedliche Größen: "
                << w1 << "x" << h1 << " vs " << w2 << "x" << h2 << std::endl;
            return 1;
        }

        int width = w1, height = h1;

        // Prüfen, ob der zu ignorierende Rand sinnvoll ist
        if (2 * ignore_border >= width || 2 * ignore_border >= height) {
            std::cerr << "Warnung: ignore_border ist zu groß, es werden möglicherweise keine Pixel verglichen." << std::endl;
        }

        long long matches = 0;
        long long total = 0;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Rand ignorieren?
                if (x < ignore_border || x >= width - ignore_border ||
                    y < ignore_border || y >= height - ignore_border) {
                    continue;
                }
                ++total;
                if (img1[y * width + x] == img2[y * width + x]) {
                    ++matches;
                }
            }
        }

        if (total == 0) {
            std::cout << "Keine Pixel im Vergleichsbereich (Rand zu groß?)." << std::endl;
            return 0;
        }

        double percent = (100.0 * matches) / total;
        std::cout << "Übereinstimmung: " << matches << " von " << total
            << " Pixeln (" << percent << "%)" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Fehler: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}