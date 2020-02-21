module Neuratron
  class RBFModel

    class Regression
        def train(model, expected_outputs : Array(Float64))
            return LibNeuratron.train(model, exepected_outputs.to_unsafe, expected_outputs.size)
        end
    end

    class Classification
        def train(model, expected_outputs : Array(Array(Float64)))
            return LibNeuratron.train(model, exepected_outputs.to_unsafe, expected_outputs.size)
        end
    end
    @model : LibNeuratron::RadialModel*

    def initialize(examples : Array(Array(Float64)), output_size : Int32, gamma : Float64)
        @model = LibNeuratron.create_radial_model(examples.flatten.to_unsafe, examples.size, examples[0].size, output_size, gamma)
    end

    def predict
    end

    def train(expected_outputs : Array(Array(Float64)), kind = Regression.new)
        kind.train(@model, expected_outputs.flatten)
    end

    def save(filename : String)
    end
  end
end
