clear all;
close all;

f=1e6
w=2*pi*f
ri=7
rds=400
cgs=.3e-12
cds=0.12e-12
gm=40e-3
zo=50

z11=ri-i/(w*cgs)
z22=1/(1/rds+i*w*cds)

s11=(z11-zo)/(z11+zo)
s22=(z22-zo)/(z22+zo)
s21=2*gm*z22*zo/(i*w*cgs)
s21=s21/(z11+zo)
s21=s21/(z22+zo)