/* To use function from matlab, first compile by entering this into the Matlab command window:
   mex LIF4pop.c
   Then call the function like this:
   s=LIF4pop(W,I0,Np,Ns,Nv,Iapp,Jee,Jep,Jes,Jev,Jpe,Jpp,Jps,Jpv,Jse,Jsp,Jss,Jsv,Jve,Jvp,Jvs,Jvv,gL,Vth,Vre,Vlb,V0,T,dt,Tburn);
 */

#include "mex.h"
#include "math.h"
#include "time.h"
#include "stdio.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

int Ne,Np,Ns,Nv,ni,typ,j,k,i,N,Nw,N2,Nt,Ntalpha,m1,m2,precell,postcell,nr,ns,maxns,jj,nX,posx;
double *I0,T,*I, Vlb,*W,*Iapp,*alpha,taualpha,*gL,taume,taump,taums,Vth,Vre,dt,*s,*J,*v,tauref,*refstate,ttemp,*lastspike,*v0,temptemp,dX,*X,Tburn;
double Jee,Jep,Jes,Jev,Jpe,Jpp,Jps,Jpv,Jse,Jsp,Jss,Jsv,Jve,Jvp,Jvs,Jvv;
mxArray *temp1, *temp2, *temp0, *temp3, *temp4;


W = mxGetPr(prhs[0]);
Nw = mxGetM(prhs[0]);
m1 = mxGetN(prhs[0]);
if(m1!=1){
    mexErrMsgTxt("Weight matrix must be Nwx1 where Nw is numer of connections.");
}
printf("Nw %d, m1 %d\n", Nw, m1);

// Check for -1 values in W
    for (i = 0; i < Nw; i++) {
        if (W[i] == -1) {
            mexErrMsgTxt("Error: Weight matrix contains -1 values.");
        }
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
maxns = ((int)mxGetScalar(prhs[29]));
printf("T %f dt %f maxns %f", T, dt, maxns);

/******
 * Finished importing variables.
 *******/


plhs[0] = mxCreateDoubleMatrix(2, maxns, mxREAL);
s=mxGetPr(plhs[0]);

temp0=mxCreateDoubleMatrix(N, 1, mxREAL);
v = mxGetPr(temp0);
printf("Allocated temp0: %p\n", (void*)temp0);

// Firing rates
dX =0.005;
nX=200;


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
  
    /* Loop over neurons */
    for(j=0;j<N;j++){

    //   printf("%d ",j) ;
      /* Update membrane potential */
      /* Spikes will be propagated at the END of the time bin (see below)*/
      v[j]+=fmax((Iapp[j]-gL[j]*v[j])*dt,Vlb-v[j]);

      /* If a spike occurs */
      if(v[j]>=Vth && ns<maxns){

          // printf("Spike from %d\n", j);


          v[j]=Vre;    /* reset membrane potential */
          s[0+2*ns]=i*dt; /* spike time */
          s[1+2*ns]=j+1;  /* neuron index */
          ns++;           /* update total number of spikes */
          //printf("Spike from j=%d\n", j);

          //printf("Spike stored\n");

          /* For all synapses for which j is the presynaptic cell index */
          for(k=(int)I0[j]-1;k<(int)I0[j+1]-1;k++){
                // if (k < 0 || k >= Nw) {
                //     printf("Error: k out of bounds. k=%d, j=%d, i=%d\n", k, j, i);
                //     printf("Nw = %d\n", Nw);
                //         mexErrMsgTxt("Index k out of bounds.");
                //     }
              postcell=((int)W[k])-1; /* This is the postsynaptic cell index */
              if (postcell < 0 || postcell >= N) {
                      printf("Error: postcell index out of bounds. postcell=%d, k=%d, j=%d, i=%d\n", postcell, k, j, i);
                      printf("W[k] = %f\n", W[k]);
                        mexErrMsgTxt("postcell index out of bounds.");
                    }

              // Pre E
              if (j<Ne && postcell<Ne)  //Post E
                v[postcell]+=Jee;
              else if (j<Ne && postcell<Ne+Np) //Post P
                v[postcell]+=Jpe;
              else if (j<Ne && postcell<Ne+Np+Ns)
                v[postcell]+=Jse;
              else if (j<Ne && postcell >= Ne+Np+Ns)
                v[postcell]+=Jve;
              else if (j<Ne+Np && postcell<Ne)
                v[postcell]+=Jep;
              else if (j<Ne+Np && postcell<Ne+Np)
                v[postcell]+=Jpp;
              else if (j<Ne+Np && postcell<Ne+Np+Ns)
                v[postcell]+=Jsp;
              else if (j<Ne+Np && postcell>=Ne+Np+Ns)
                v[postcell]+=Jvp;
              else if (j<Ne+Np+Ns && postcell<Ne)
                v[postcell]+=Jes;
              else if (j<Ne+Np+Ns && postcell<Ne+Np)
                v[postcell]+=Jps;
              else if (j<Ne+Np+Ns && postcell<Ne+Np+Ns)
                v[postcell]+=Jss;
              else if (j<Ne+Np+Ns && postcell>=Ne+Np+Ns)
                v[postcell]+=Jvs;
              else if (postcell<Ne)
                v[postcell]+=Jev;
              else if (postcell<Ne+Np)
                v[postcell]+=Jpv;
              else if (postcell<Ne+Np+Ns)
                v[postcell]+=Jsv;
              else if (postcell>=Ne+Np+Ns)
                v[postcell]+=Jvv;
              else
                mexErrMsgTxt("Error in connectivity.");
              }

          }
      }
      /* Check if i is a multiple of 1/dt and print progress */
        if(i%(int)(1/dt)==0){
            mexPrintf("Time: %f\n",i*dt);
//            mexEvalString("drawnow;");
            }
    }


printf("Exit Time Loop.\n");

/* Issue a warning if max number of spikes reached */
if(ns>=maxns)
   mexWarnMsgTxt("Maximum number of spikes reached, simulation terminated.");

printf("done.\n");

printf("Before mxDestroyArray(temp0): %p\n", (void*)temp0);
mxDestroyArray(temp0);
printf("After mxDestroyArray(temp0)\n");

printf("done2.\n");

}
