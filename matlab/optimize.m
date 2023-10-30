load("optimize.mat")
options = optimoptions('quadprog','Display','iter');
lb = [];
ub = [];
[x fval,exitflag,output] = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
[argvalue, argmax] = max(x);