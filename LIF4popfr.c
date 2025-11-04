/* To use function from matlab, first compile by entering this into the Matlab command window:
   mex LIF4popfr.c
   Then call the function like this:
   frates=LIF4popfr(W,I0,Np,Ns,Nv,Iapp,Jee,Jep,Jes,Jev,Jpe,Jpp,Jps,Jpv,Jse,Jsp,Jss,Jsv,Jve,Jvp,Jvs,Jvv,gL,Vth,Vre,Vlb,V0,T,dt,Tburn);
 */

#include "mex.h"
#include "math.h"
#include "time.h"
#include "stdio.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

int Ne,Np,Ns,Nv,ni,typ,j,k,i,N,Nw,N2,Nt,Ntalpha,m1,m2,precell,postcell,nr,ns,jj,nX,posx;
double *I0,T,*I, Vlb,*W,*Iapp,*alpha,taualpha,*gL,taume,taump,taums,Vth,Vre,dt,*frates,*J,*v,tauref,*refstate,ttemp,*lastspike,*v0,temptemp,dX,*X,Tburn;
double Jee,Jep,Jes,Jev,Jpe,Jpp,Jps,Jpv,Jse,Jsp,Jss,Jsv,Jve,Jvp,Jvs,Jvv;
double enorm,pnorm,snorm,vnorm;
mxArray *temp1, *temp2, *temp0, *temp3, *temp4;

W = mxGetPr(prhs[0]);
Nw = mxGetM(prhs[0]);
m1 = mxGetN(prhs[0]);
if(m1!=1){
    mexErrMsgTxt("Weight matrix must be Nwx1 where Nw is numer of connections.");
}


I0 = mxGetPr(prhs[1]);
N = mxGetM(prhs[1])-1;
m2 = mxGetN(prhs[1]);
if(N==1 && m2!=1){
    N=m2;
    m2=1;
}
if(N==1 && m2==1){
    mexErrMsgTxt("I0 should be Nx1.");
}
if(N!=1 && m2!=1){
    mexErrMsgTxt("I0 should be Nx1..");
}

Np=(int)mxGetScalar(prhs[2]);
Ns=(int)mxGetScalar(prhs[3]);
Nv=(int)mxGetScalar(prhs[4]);
Ne = N - Np - Ns - Nv;
printf("Ne %d Np %d Ns %d Nv %d N %d\n", Ne, Np, Ns, Nv, N);

Iapp = mxGetPr(prhs[5]);
m1 = mxGetM(prhs[5]);
m2 = mxGetN(prhs[5]);
if(m1!=N && m2!=N){
    mexErrMsgTxt("Iapp should be Nx1.");
}
if(m1!=1 && m2!=1){
    mexErrMsgTxt("Iapp should be Nx1.");
}


Jee= mxGetScalar(prhs[6]);
Jep= mxGetScalar(prhs[7]);
Jes= mxGetScalar(prhs[8]);
Jev= mxGetScalar(prhs[9]);
Jpe= mxGetScalar(prhs[10]);
Jpp= mxGetScalar(prhs[11]);
Jps= mxGetScalar(prhs[12]);
Jpv= mxGetScalar(prhs[13]);
Jse= mxGetScalar(prhs[14]);
Jsp= mxGetScalar(prhs[15]);
Jss= mxGetScalar(prhs[16]);
Jsv= mxGetScalar(prhs[17]);
Jve= mxGetScalar(prhs[18]);
Jvp= mxGetScalar(prhs[19]);
Jvs= mxGetScalar(prhs[20]);
Jvv= mxGetScalar(prhs[21]);

printf("Jee %f Jep %f Jes %f Jev %f\n", Jee, Jep, Jes, Jev);
printf("Jpe %f Jpp %f Jps %f Jpv %f\n", Jpe, Jpp, Jps, Jpv);
printf("Jse %f Jsp %f Jss %f Jsv %f\n", Jse, Jsp, Jss, Jsv);
printf("Jve %f Jvp %f Jvs %f Jvv %f\n", Jve, Jvp, Jvs, Jvv);

gL = mxGetPr(prhs[22]);
m1 = mxGetM(prhs[22]);
m2 = mxGetN(prhs[22]);
if(m1!=N && m2!=N){
    mexErrMsgTxt("gL should be Nx1.");
}
if(m1!=1 && m2!=1){
    mexErrMsgTxt("gL should be Nx1.");
}


Vth = mxGetScalar(prhs[23]);
Vre = mxGetScalar(prhs[24]);
Vlb = mxGetScalar(prhs[25]);

printf("Vth %f Vre %f Vlb %f\n", Vth, Vre, Vlb);

v0 = mxGetPr(prhs[26]);
m1 = mxGetM(prhs[26]);
m2 = mxGetN(prhs[26]);
if(!((m1==1&&m2==N)||(m1==N&&m2==1))){
    mexErrMsgTxt("V0 should be Nx1.");
}
T = mxGetScalar(prhs[27]);
dt = mxGetScalar(prhs[28]);
Tburn = mxGetScalar(prhs[29]);
/******
 * Finished importing variables.
 *******/

printf("T = %f dt = %f Tburn = %f", T, dt, Tburn);

/******
 * Now allocate new variables.
 *****/

temp0=mxCreateDoubleMatrix(N, 1, mxREAL);
v = mxGetPr(temp0);

// Firing rates
dX =0.005;
nX=200;

plhs[0] = mxCreateDoubleMatrix(4, nX, mxREAL);
frates = mxGetPr(plhs[0]);
printf("Created frates");

/*****
 * Finished allocating variables
 ****/



/* Number of time bins */
Nt=(int)(T/dt);


/* Inititalize v */
for(j=0;j<N;j++){
    v[j]=Iapp[j]*dt+v0[j];
}

/* Initialize number of spikes */
ns=0;

printf("Starting loop\n");

/* Time loop */
/* Exit loop and issue a warning if max number of spikes is exceeded */
for(i=1;i<Nt;i++){
    /* printf("t = %d\n", i*dt);
    /* Loop over neurons */
    for(j=0;j<N;j++){

    //   printf("%d ",j) ;
      /* Update membrane potential */
      /* Spikes will be propagated at the END of the time bin (see below)*/
      v[j]+=fmax((Iapp[j]-gL[j]*v[j])*dt,Vlb-v[j]);

      /* If a spike occurs */
      if(v[j]>=Vth){

        //   printf("Spike from %d\n", j);

          v[j]=Vre;    /* reset membrane potential */
          //s[0+2*ns]=i*dt; /* spike time */
          //s[1+2*ns]=j+1;  /* neuron index */
          ns++;           /* update total number of spikes */
          // printf("Spike from j=%d\n", j);

          typ = 0;

          if (j < Ne){
            typ = 0;
            ni = j;
            posx = (int)floor((200*ni)/Ne);
          }
          else if (j < Ne+Np){
            typ = 1;
            ni = j - Ne;
            posx = (int)floor((200*ni)/Np);
          }
          else if (j < Ne+Np+Ns){
            typ = 2;
            ni = j - Ne - Np;
            posx = (int)floor((200*ni)/Ns);
          }
          else{
            typ = 3;
            ni = j - Ne - Np - Ns;
            posx = (int)floor((200*ni)/Nv);
          }

          if(i*dt>Tburn){
              frates[typ + posx * 4] += 1.0;
          }
          //printf("Spike stored\n");

          /* For all synapses for which j is the presynaptic cell index */
          for(k=(int)I0[j]-1;k<(int)I0[j+1]-1;k++){
              postcell=((int)W[k])-1; /* This is the postsynaptic cell index */
              if(typ==0){ // Pre E
                if (postcell<Ne)  //Post E
                  v[postcell]+=Jee;
                else if (postcell<Ne+Np) //Post P
                  v[postcell]+=Jpe;
                else if (postcell<Ne+Np+Ns)
                  v[postcell]+=Jse;
                else
                  v[postcell]+=Jve;
              }
              else if(typ==1){
                if (postcell<Ne)
                  v[postcell]+=Jep;
                else if (postcell<Ne+Np)
                  v[postcell]+=Jpp;
                else if (postcell<Ne+Np+Ns)
                  v[postcell]+=Jsp;
                else
                  v[postcell]+=Jvp;
              }
              else if(typ==2){
                if (postcell<Ne)
                  v[postcell]+=Jes;
                else if (postcell<Ne+Np)
                  v[postcell]+=Jps;
                else if (postcell<Ne+Np+Ns)
                  v[postcell]+=Jss;
                else
                  v[postcell]+=Jvs;
              }
              else if(typ==3){
                if (postcell<Ne)
                  v[postcell]+=Jev;
                else if (postcell<Ne+Np)
                  v[postcell]+=Jpv;
                else if (postcell<Ne+Np+Ns)
                  v[postcell]+=Jsv;
                else
                  v[postcell]+=Jvv;
              }
              else
                  mexErrMsgTxt("Error in connectivity.");
              }

              /* Print pre and post spike index */
              //if (6998< postcell < 7002)
              //printf("Pre %d Post %d\n", j, postcell);
          }
      }
      /* Check if i is a multiple of 1/dt and print progress */
        if(i%(int)(1/dt)==0){
            printf("Time: %f\n",i*dt);
//            mexEvalString("drawnow;");
            }
    }

enorm = 1000*200/(Ne*(T-Tburn));
pnorm = 1000*200/(Np*(T-Tburn));
snorm = 1000*200/(Ns*(T-Tburn));
vnorm = 1000*200/(Nv*(T-Tburn));



for(i=0;i<nX;i++){
    frates[0 + i*4] = frates[0 + i*4]*enorm;
    frates[1 + i*4] = frates[1 + i*4]*pnorm;
    frates[2 + i*4] = frates[2 + i*4]*snorm;
    frates[3 + i*4] = frates[3 + i*4]*vnorm;
}

printf("Exit Time Loop.\n");

/* Issue a warning if max number of spikes reached */
//if(ns>=maxns)
//   mexWarnMsgTxt("Maximum number of spikes reached, simulation terminated.");

printf("done.\n");

/* Free allocated memory */
mxDestroyArray(temp0);

printf("done2.\n");


}
