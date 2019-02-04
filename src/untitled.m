clear all;
close all;

X= [1.1, 1.0, 1.3,1;
    1.1, 1.2, 1.2,1;
    3.0, 3.0, 3.1,1;
    3.5, 3.3, 3.2,1];%here last 1 in every row for bias
Y = [1,1,0,0];
alpha = 0.1;
w=rand(4,1);
for e=1:100
    for i=1:4
        x=X(i,:);
        y=Y(i);
        
        
        y_das = 1/(1+exp(-x*w));
        
        e = .5*(y-y_das).^2;
        del = (y-y_das).*x;
        w=w-(alpha*del)';
    end
end
disp(w)
y_pred = 1/(1+exp(-X*w))

x=[3 1 1;6 2 1;2 5 1]%;5 3 1];
y=[1;1;-1]
w=inv(x)*y


