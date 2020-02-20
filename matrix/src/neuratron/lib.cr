@[Link("neuratron")]
lib LibNeuratron
  @[Extern]
  struct DeepModel
    layer_count : Int32
    deltas : Float64**
    x : Float64**
    w : Float64***
    d : Int32*
  end

  fun create_deep_model(neurons_per_layers : Int32*, size : Int32) : DeepModel*
  fun train_deep_regression_model(
    model : DeepModel*,
    input : Float64*, input_size : Int32,
    output : Float64*, output_size : Int32,
    training_rate : Float64
  ) : Bool
  fun train_deep_classification_model(
    model : DeepModel*,
    input : Float64*, input_size : Int32,
    output : Float64*, output_size : Int32,
    training_rate : Float64
  ) : Bool
  fun predict_deep_model_regression(model : DeepModel*, input : Float64*) : Float64*
  fun predict_deep_model_classification(model : DeepModel*, input : Float64*) : Float64
  fun free_deep_model(model : DeepModel*) : Int32

  @[Extern]
  struct LinearModel
    inputs: Float64*
    size_input: Int32
    size_output: Int32
  end

  fun create_linear_model(input : Int32, size: Int32) : Pointer(LinearModel)
  fun train_linear_model(
    model : LinearModel*,
    input : Float64*, input_size : Int32,
    output : Float64*, output_size : Int32
  ) : Bool

  fun predict_linear_model_regression(model : LinearModel*, input : Float64*) : Float64
  fun predict_linear_model_classification(model : LinearModel*, input : Float64*) : Float64
  fun free_linear_model(model : Pointer(LinearModel)) : Int32

  @[Extern]
  struct RadialModel
      example_count : Int32
      size_input: Int32
      size_output: Int32
      w : Float64***
      gamma: Float64
  end
  fun create_radial_model(input : Int32, size: Int32, gamma : Float64, example_count : Int32) : Pointer(RadialModel)
end
