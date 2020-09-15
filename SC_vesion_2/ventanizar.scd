//función para determinar el ventaneo
~ventanizar = {|len, arr|
	var totalFrames = (arr.size/len).asInteger;
	var shape = (1..len)!totalFrames;
	var ventanas = arr.reshapeLike(shape);
	ventanas;
};