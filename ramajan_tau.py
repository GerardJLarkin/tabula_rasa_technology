#   Copyright (C) 2016 Michael Reed

print("\nRamanujan's tau function (100 years of):")
print("The following program uses turtle to plot prime values of the Ramanujan tau\nfunction normalized by Ramanujan's third conjecture. The lines are plotted\nusing the inverse cosine angle direction and normalized length.\n")
print('In fast mode, only the statistical angle distribution is displayed.\n')

print("Setup: press <enter> for defaults...\n")
pup = input('Evaluate tau(p) using primes up to: ')
if pup == '':pup = 12345
else:pup = int(pup)
fastmode = input('Enable fast statistical mode (y/n): ')
if fastmode == 'y':fastmode = True
else:fastmode = False

# sum of diviors function
sigl = { }
def sigma(n):
    if n not in sigl:
        sigl[n] = n+ sum( [ d for d in range(1,math.ceil((n+1)/2)) if n%d==0 ] )
    return sigl[n]

# Ramanujan tau function based on https://projecteuclid.org/download/pdf_1/euclid.ijm/1256050746
def tau(p):return (p**4)*sigma(p) - 24*sum( ( (35*k**4 - 52*p*k**3 + 18*(p**2)*(k**2) )*sigma(k)*sigma(p-k) ) for k in range(1,p))

# angle of tau(p) by Ramanujan's third conjecture
import math
def theta(p):
    tp = tau(p)/(2*p**(11/2))
    return math.degrees(math.acos(tp)),tp

# angle distribution function
def angpdf(x):return math.sin(x)**2

# length normalization distribution conjecture
def lenpdf(x):return 1-x**2

# quickly generate primes list https://stackoverflow.com/questions/2068372/#2068412
import itertools
def erat2( ):
    D = {  }; yield 2
    for q in itertools.islice(itertools.count(3), 0, None, 2):
        p = D.pop(q, None)
        if p is None:D[q*q] = q; yield q
        else:
            x = p + q
            while x in D or not (x&1): x += p
            D[x] = p
def get_primes_erat(n):
    return list(itertools.takewhile(lambda p: p<n, erat2()))
pl = get_primes_erat(pup)
print("\nFound "+str(len(pl))+" primes to evaluate (computing...)\n")

# turtle results for all primes in list
count = { }; grow = {0:0}
part = round(len(pl)/(1.618**math.log(len(pl))))
angle=180;txt="Ramanujan tau function: "
for n in range(1,part+1):
    count[n]=0;grow[n]=0
def drawBar(t,height,width):
    t.left(90)
    t.forward(height)
    t.right(90)
    t.forward(width)
    t.right(90)
    t.forward(height)
    t.left(90)
div = round(len(pl)/part)
divc = 0;i=0;prec=3
if not fastmode :
    import turtle
    turtle.title(txt+"prime conjecture angle dilation normalization")
    bd=0.25;turtle.setworldcoordinates(-bd,0.5,1,-0.5)
    turtle.speed(speed=0)
    bars = turtle.Turtle();bars.speed(speed=0);bars.hideturtle()
    bdrs = turtle.Turtle();bdrs.speed(speed=0);bdrs.hideturtle()
    pmax = max(pl);turtle.shape("turtle")
    turtle.pencolor("gray");turtle.bgcolor("black")
    bars.pencolor("white");bdrs.pencolor("white")
    turtle.penup();turtle.sety(-0.5);turtle.pendown()
    for n in range(1,(part*prec)+1):
        loc=n/(part*prec);het=angpdf(math.pi*loc)
        turtle.setpos(-bd*het,-0.5+loc)
    turtle.penup();turtle.home();turtle.sety(-0.15);turtle.pendown()
    for n in range(1,(part*prec)+1):
        loc=n/(part*prec);het=lenpdf(loc)
        turtle.setpos(loc,-bd*het+0.1)
    turtle.penup();turtle.home();turtle.pendown()
    for p in pl:
        deg,dist=theta(p);i+=1
        count[round(deg*part/angle)] += 1
        grow[round(abs(dist)*part)] += 1
        turtle.left(deg)
        turtle.forward(dist)
        turtle.penup()
        turtle.home()
        turtle.pendown()
        if i == 4*div or p == pmax :
            i = 0
            maxval=count[max(count,key=count.get)]
            mbxval=grow[max(grow,key=grow.get)]
            bars.home();bdrs.home()
            bars.sety(-0.5);bdrs.sety(0.1);
            bars.left(90);
            bars.clear();bdrs.clear();
            for n in range(1,part+1):
                drawBar(bdrs,-bd*grow[n]/mbxval,1/part)
                drawBar(bars,bd*count[n]/maxval,1/part)
    print('Done.')
else:
    for p in pl:
        deg,dist=theta(p);i+=1
        count[round(abs(deg)*part/angle)] += 1
        grow[round(abs(dist)*part)] +=1
        if i == div:
            i = 0; divc += 1;
            print("Evaluating: "+str(divc*div)+"/"+str(len(pl)))
    maxval=count[max(count,key=count.get)]
    mbxval=grow[max(grow,key=grow.get)]
    print('Done.')
    import turtle
    turtle.title(txt+"Sato-Tate angle dilation statistical distribution")
    turtle.setworldcoordinates(0,-0.36,1,0.64);turtle.speed(speed=0)
    for n in range(1,part+1):
        drawBar(turtle,0.59*count[n]/maxval,1/part)
    turtle.home()
    for n in range(1,(part*prec)+1):
        loc=n/(part*prec);het=angpdf(math.pi*loc)
        turtle.setpos(loc,0.59*het)
    turtle.home();turtle.penup();turtle.sety(-0.36);turtle.pendown()
    for n in range(0,part+1):
        drawBar(turtle,0.31*grow[n]/mbxval,1/(part+1))
    turtle.setx(0);turtle.penup();turtle.sety(-0.05);turtle.pendown()
    for n in range(1,(part*prec)+1):
        loc=n/(part*prec);het=lenpdf(loc)
        turtle.setpos(loc,0.31*het-0.36)
turtle.hideturtle()
turtle.done()