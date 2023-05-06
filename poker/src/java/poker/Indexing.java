package poker;
import java.util.Random;

public class Indexing {
    public static float[] indexIntoBlock(int iStart, int[] indices, float[] block, int n, float stdev){
        float[] res = new float[n];
        int N = block.length;

        int[] indicesTracker = new int[indices.length];
        for (int i=0;i<indices.length;i++){
            indices[i] = ((indices[i]% N)+N)%N;
            indicesTracker[i] = ((indices[i]*iStart % N) + N) % N;
        }

        for (int i=0;i<n;i++){
            res[i] = addIndices(indicesTracker, indices, block, N, stdev);
            
        }
        return res;
    }

    public static float addIndices(int[] indicesTracker ,int[] indices, float[] block, int N, float stdev){
        int n = indices.length;
        float res = 0;
        for (int i=0;i<n;i++){
            res+=stdev*block[indicesTracker[i]];
            indicesTracker[i]=(indicesTracker[i]+indices[i])%N;
            //res+=stdev*block[((indices[i]*iStart % N)+N)%N];
        }
        return res;
    }

    public static float[] initializeRandomBlock(int n, Random r){
        float[] res = new float[n];
        for (int i=0;i<n;i++) res[i] = (float) r.nextGaussian();
        return res;
    }
}
