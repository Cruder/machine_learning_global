@[Link("neuratron")]
lib LibNeuratron
  @[Extern]
  struct LinearModel
    inputs: Float64*
    size_input: Int32
    size_output: Int32
  end

  fun create_linear_model(input : Int32, size: Int32) : Pointer(LinearModel)
  fun train_linear_model = train_linear_model(i : Int32) : Int32
end


module Neuratron
  class LinearModel
    @model : LibNeuratron::LinearModel*

    def initialize(@input : Int32, @ouput : Int32)
      @model = LibNeuratron.create_linear_model(input, ouput)
    end

    def model
      @model.value
    end
  end
end

inputs = [
  [1.0, 2.0],
  [3.0, 4.0],
  [5.0, 5.0],
  # [6.0, 6.0],
]
expected_outputs = [
  [2],
  [3],
  [1],
  # [40],
]

model = Neuratron::LinearModel.new(2, 1)
pp model.model.size_input
pp model.model.size_output
