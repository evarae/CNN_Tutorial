package network;

import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import layers.MaxPoolLayer;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {

    private NeuralNetwork net;
    private int _inputRows;
    private int _inputCols;
    private double _scaleFactor;
    List<Layer> _layers;

    public NetworkBuilder(int _inputRows, int _inputCols, double _scaleFactor) {
        this._inputRows = _inputRows;
        this._inputCols = _inputCols;
        this._scaleFactor = _scaleFactor;
        _layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long SEED){
        if(_layers.isEmpty()){
            _layers.add(new ConvolutionLayer(filterSize, stepSize, 1, _inputRows, _inputCols, SEED, numFilters, learningRate));
        } else {
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), SEED, numFilters, learningRate));
        }
    }

    public void addMaxPoolLayer(int windowSize, int stepSize){
        if(_layers.isEmpty()){
            _layers.add(new MaxPoolLayer(stepSize, windowSize, 1, _inputRows, _inputCols));
        } else {
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    public void addFullyConnectedLayer(int outLength, double learningRate, long SEED){
        if(_layers.isEmpty()) {
            _layers.add(new FullyConnectedLayer(_inputCols*_inputRows, outLength, SEED, learningRate));
        } else {
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new FullyConnectedLayer(prev.getOutputElements(), outLength, SEED, learningRate));
        }

    }

    public NeuralNetwork build(){
        net = new NeuralNetwork(_layers, _scaleFactor);
        return net;
    }

}
