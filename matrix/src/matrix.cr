require "aquaplot"

@[Link("neuratron")]
lib LibNeuratron
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

  fun predict_linear_model(model : LinearModel*, input : Float64*) : Float64
  fun free_model(model : Pointer(LinearModel)) : Int32
end

module Neuratron
  class LinearModel
    @model : LibNeuratron::LinearModel*

    def initialize(@input_size : Int32, @ouput_size : Int32)
      @model = LibNeuratron.create_linear_model(input_size, ouput_size)
    end

    def finalize
      LibNeuratron.free_model(@model)
    end

    def model
      @model.value
    end

    def weights
      (0...((model.size_input + 1) * model.size_output)).map do |i|
        model.inputs[i]
      end
    end

    def train(input, output)
      LibNeuratron.train_linear_model(@model, input.to_unsafe, input.size, output.to_unsafe, output.size)
    end

    def predict(input)
      LibNeuratron.predict_linear_model(@model, input.to_unsafe)
    end
  end
end

inputs = [
  [1.0, 2.0],
  [3.0, 4.0],
  [5.0, 5.0],
  [6.0, 6.0],
]
expected_outputs = [
  [2.0],
  [3.0],
  [1.0],
  [40.0],
]

model = Neuratron::LinearModel.new(2, 1)
model.train(inputs.flatten, expected_outputs.flatten)

predictions = inputs.map do |input|
  puts "Predict for #{input}"
  results = model.predict(input)
  puts "Prediction #{results}"
  results
end

positions = inputs.zip(predictions).map do |data|
  { data[0][0], data[0][1],  data[1] }
end

pp "positions", positions

math_formulat = "#{model.weights[0]} * 1 + #{model.weights[1]} * x + #{model.weights[2]} * y"
fns = [
  AquaPlot::Scatter3D.from_points(positions).tap(&.set_title("Points")),
  AquaPlot::Function.new("0", title: "0"),
  AquaPlot::Function.new(math_formulat, title: math_formulat),
]

pp fns[-1].style = "pm3d"

plt = AquaPlot::Plot3D.new fns
plt.set_view(100, 80, 1)
plt.set_key("left box")
plt.show
plt.close
