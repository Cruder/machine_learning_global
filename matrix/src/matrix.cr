require "aquaplot"
require "json"
require "./neuratron/lib"
require "./neuratron/**"
require "./mnist/loader"

def max_indice(array : Array(Float64))
  max = array[0];
  index = 0;
  array.each_with_index do |e, i|
    if e > max
      max = e
      index = i
    end
  end
  index
end

def clamp_value_rgb(x)
  x / 256.0 * 2 - 1
end

def print_mnist_ascii(image)
  image.each_with_index do |pixel, i|
    print pixel < 0.3 ? "." : "#"
    puts if i % 28 == 0 && i != 0
  end
  puts
end
