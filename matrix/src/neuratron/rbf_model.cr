module Neuratron
  class RBFModel

    class Regression
        def train(model, expected_outputs : Array(Float64), iteration : Int32)
            return LibNeuratron.train_radial_regression(model, expected_outputs.to_unsafe)
        end
        def predict(model, inputs : Array(Float64))
            LibNeuratron.predict_radial_regression(model, inputs.to_unsafe, inputs.size)
        end
    end

    class Classification
        def train(model, expected_outputs : Array(Float64), iteration : Int32)
            LibNeuratron.train_radial_classification(model, expected_outputs.to_unsafe, iteration)
        end
        def predict(model, inputs : Array(Float64))
            LibNeuratron.predict_radial_classification(model, inputs.to_unsafe, inputs.size)
        end
    end
    @model : LibNeuratron::RadialModel*

    def initialize(examples : Array(Array(Float64)), output_size : Int32, gamma : Float64)
        @model = LibNeuratron.create_radial_model(examples.flatten.to_unsafe, examples.size, examples[0].size, output_size, gamma)
        #pp @model.value
    end

    def predict(inputs : Array(Array(Float64)), kind = Regression.new)
        kind.predict(@model, inputs.flatten)
    end

    def train(expected_outputs : Array(Array(Float64)), iteration : Int32, kind = Regression.new)
        kind.train(@model, expected_outputs.flatten, iteration)
    end

    def save(filename : String)
    end
  end
end
