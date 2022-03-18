T = readtable('train.csv');
y_vector = T.y;
x_matrix = [T.x1, T.x2 T.x3 T.x4 T.x5];

a = 1 + zeros( 1, 700 );

A_matrix = [x_matrix, x_matrix.*x_matrix, exp(x_matrix), cos(x_matrix), a' ];

x=lsqr(A_matrix(21:700, 1:21), y_vector(21:700));

writematrix(x, 'matlab_result.csv')