package week_02;

import java.io.*;
import java.nio.*;

// ---------------- Config ----------------
class Config {
    int dim;        // transformer dimension
    int hiddenDim;  // hidden dimension for ffn
    int nLayers;    // number of layers
    int nHeads;     // number of query heads
    int nKvHeads;   // number of key/value heads
    int vocabSize;  // vocab size (absolute value if negative)
    int seqLen;     // max sequence length
}

// ---------------- Transformer Weights ----------------
class TransformerWeights {
    float[] tokenEmbeddingTable; // (vocab_size, dim)
    float[] rmsAttWeight;        // (layer, dim)
    float[] rmsFfnWeight;        // (layer, dim)
    float[] wq, wk, wv, wo;      // attention weights
    float[] w1, w2, w3;          // feed-forward weights
    float[] rmsFinalWeight;      // (dim,)
    float[] wcls;                // classifier / shared embedding
}

// ---------------- Checkpoint Loader ----------------
public class llama2 {

    // Read int in little-endian format
    private static int readIntLE(DataInputStream dis) throws IOException {
        byte[] b = new byte[4];
        dis.readFully(b);
        return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).getInt();
    }

    // Read float in little-endian format
    private static float readFloatLE(DataInputStream dis) throws IOException {
        byte[] b = new byte[4];
        dis.readFully(b);
        return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).getFloat();
    }

    // Read float array
    private static float[] readFloatArray(DataInputStream dis, int size) throws IOException {
        float[] arr = new float[size];
        for (int i = 0; i < size; i++) {
            arr[i] = readFloatLE(dis);
        }
        return arr;
    }

    // Skip floats
    private static void skipFloats(DataInputStream dis, int count) throws IOException {
        for (int i = 0; i < count; i++) {
            readFloatLE(dis);
        }
    }

    public static void readCheckpoint(String path, Config config, TransformerWeights weights) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {

            // ---------- 1. Read Config (7 ints, little-endian) ----------
            config.dim       = readIntLE(dis);
            config.hiddenDim = readIntLE(dis);
            config.nLayers   = readIntLE(dis);
            config.nHeads    = readIntLE(dis);
            config.nKvHeads  = readIntLE(dis);
            config.vocabSize = readIntLE(dis);
            config.seqLen    = readIntLE(dis);

            // vocabSize < 0 means no shared weights
            boolean sharedWeights = config.vocabSize > 0;
            config.vocabSize = Math.abs(config.vocabSize);

            int headSize = config.dim / config.nHeads;

            // ---------- 2. Read weights in order ----------
            weights.tokenEmbeddingTable = readFloatArray(dis, config.vocabSize * config.dim);
            weights.rmsAttWeight        = readFloatArray(dis, config.nLayers * config.dim);
            weights.wq                  = readFloatArray(dis, config.nLayers * config.dim * (config.nHeads * headSize));
            weights.wk                  = readFloatArray(dis, config.nLayers * config.dim * (config.nKvHeads * headSize));
            weights.wv                  = readFloatArray(dis, config.nLayers * config.dim * (config.nKvHeads * headSize));
            weights.wo                  = readFloatArray(dis, config.nLayers * (config.nHeads * headSize) * config.dim);
            weights.rmsFfnWeight        = readFloatArray(dis, config.nLayers * config.dim);
            weights.w1                  = readFloatArray(dis, config.nLayers * config.dim * config.hiddenDim);
            weights.w2                  = readFloatArray(dis, config.nLayers * config.hiddenDim * config.dim);
            weights.w3                  = readFloatArray(dis, config.nLayers * config.dim * config.hiddenDim);
            weights.rmsFinalWeight      = readFloatArray(dis, config.dim);

            // ---------- 3. Skip RoPE freq_cis_real and freq_cis_imag ----------
            skipFloats(dis, config.seqLen * headSize);

            // ---------- 4. Classifier weights ----------
            if (sharedWeights) {
                weights.wcls = weights.tokenEmbeddingTable;
            } else {
                weights.wcls = readFloatArray(dis, config.vocabSize * config.dim);
            }
        }
    }

    // ---------------- Test Entry Point ----------------
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java week_02.llama2 <checkpoint.bin>");
            return;
        }

        String checkpointPath = args[0];
        Config cfg = new Config();
        TransformerWeights w = new TransformerWeights();

        readCheckpoint(checkpointPath, cfg, w);

        // Print results for verification
        System.out.println("Config:");
        System.out.println("  dim       = " + cfg.dim);
        System.out.println("  hiddenDim = " + cfg.hiddenDim);
        System.out.println("  nLayers   = " + cfg.nLayers);
        System.out.println("  nHeads    = " + cfg.nHeads);
        System.out.println("  nKvHeads  = " + cfg.nKvHeads);
        System.out.println("  vocabSize = " + cfg.vocabSize);
        System.out.println("  seqLen    = " + cfg.seqLen);

        System.out.println("Weights:");
        System.out.println("  tokenEmbeddingTable length = " + w.tokenEmbeddingTable.length);
        System.out.println("  rmsFinalWeight[0]         = " + w.rmsFinalWeight[0]);
    }
}
