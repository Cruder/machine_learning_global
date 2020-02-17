@[Link("neuratron")]
lib LibLibrary
  @[Extern]
  struct Foo
    array : Pointer(Int32)
    size : Int32
  end

  fun bar = init_bar() : Int32
  fun foo = init_foo() : Pointer(Foo)
  fun give_me_42 = give_42() : Int32*
end

class Giver42
  @value : Int32
  @value2 : LibLibrary::Foo
  @value3 : Int32

  def initialize
    @value = LibLibrary.give_me_42.value
    @value2 = LibLibrary.foo.value
    @value3 = LibLibrary.bar
  end

  def give
    @value
  end

  def give2
    @value2
  end

  def give3
    @value3
  end
end

pp LibLibrary.give_me_42.value

pp Giver42.new.give

foo = Giver42.new.give3
pp foo


bar = Giver42.new.give2
pp bar.size
pp bar.array
