package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{

    private long SEED;
    private final double leak = 0.01;

    private double[][] _weights;
    private int _inLength;
    private int _outLength;
    private double _learningRate;

    private double[] lastZ;
    private double[] lastX;


    public FullyConnectedLayer(int _inLength, int _outLength, long SEED, double learningRate) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;
        this._learningRate = learningRate;

        _weights = new double[_inLength][_outLength];
        setRandomWeights();
    }

    public double[] fullyConnectedForwardPass(double[] input){

        lastX = input;

        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        for(int i = 0; i < _inLength; i++){
            for(int j = 0; j < _outLength; j++){
                z[j] += input[i]*_weights[i][j];
            }
        }

        lastZ = z;

        for(int i = 0; i < _inLength; i++){
            for(int j = 0; j < _outLength; j++){
                out[j] = reLu(z[j]);
            }
        }

        return out;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);

        if(_nextLayer != null){
            return _nextLayer.getOutput(forwardPass);
        } else {
            return forwardPass;
        }
    }

    @Override
    public void backPropagation(double[] dLdO) {

        double[] dLdX = new double[_inLength];

        double dOdz;
        double dzdw;
        double dLdw;
        double dzdx;

        for(int k = 0; k < _inLength; k++){

            double dLdX_sum = 0;

            for(int j = 0; j < _outLength; j++){

                dOdz = derivativeReLu(lastZ[j]);
                dzdw = lastX[k];
                dzdx = _weights[k][j];

                dLdw = dLdO[j]*dOdz*dzdw;

                _weights[k][j] -= dLdw*_learningRate;

                dLdX_sum += dLdO[j]*dOdz*dzdx;
            }

            dLdX[k] = dLdX_sum;
        }

        if(_previousLayer!= null){
            _previousLayer.backPropagation(dLdX);
        }
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outLength;
    }

    public void setRandomWeights(){
        Random random = new Random(SEED);

        for(int i = 0; i < _inLength; i++){
            for(int j =0; j < _outLength; j++){
                _weights[i][j] = random.nextGaussian();
            }
        }
    }

    public double reLu(double input){
        if(input <= 0){
            return 0;
        } else {
            return input;
        }
    }

    public double derivativeReLu(double input){
        if(input <= 0){
            return leak;
        } else {
            return 1;
        }
    }

}
