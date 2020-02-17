@[Link("neuratron")]
lib LibLibrary
  struct Foo
    array : Pointer(Int32)
    size : Int32
  end

  fun give_me_42 = give_42() : Int32*
  fun foo = init_foo() : Foo*
end

class Giver42
  @value : Int32

  def initialize
    @value = LibLibrary.give_me_42.value
  end

  def give
    @value
  end
end

pp LibLibrary.give_me_42.value

pp Giver42.new.give

foo = LibLibrary.foo
pp foo.value.size
