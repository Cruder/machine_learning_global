@[Link("neuratron")]
lib LibNeuratron
  @[Extern]
  struct LinearModel
    inputs: Float64*
    size_inputs: Int32
  end

  fun create_linear_model(inputs : Float64*, size: Int32) : Pointer(LinearModel)
  fun train_linear_model = train_linear_model(i : Int32) : Int32
end

class TronLinearModel
  @model: LibNeuratron::LinearModel

  def initialize(inputs : Array(Float64))
    @model = LibNeuratron.create_linear_model(inputs, inputs.size).value
  end

  def train
    LibNeuratron.train_linear_model(10)
  end
end
inputs = [  0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0  ]
expected_outputs = [0, 0, 1, 1]
model = TronLinearModel.new(inputs)
pp model.train
