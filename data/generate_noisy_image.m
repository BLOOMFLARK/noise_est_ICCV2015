rand( 'seed', 0 );
randn( 'seed', 0 );

x = double( imread('cameraman.tif') );
y = x + 5.0 * randn(size(x));
y = max( 0, min(y,255) );
y = uint8(y);

imwrite( y, 'cameraman-std=5.pgm' );
