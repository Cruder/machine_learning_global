require "./spec_helper"

describe Neuratron::DeepModel do
  it "with 4 exemples" do
    # model = Neuratron::DeepModel.new(2, 5, 1)
    # result = model.predict([1.0, 2.0])
    # pp "result", result
    # pp "model", model
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

    model = Neuratron::DeepModel.new(2,3,1)
    model.train_regression(inputs, expected_outputs, 2, 0.4)
  end
end
