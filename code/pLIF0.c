#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    int Nv, k;
    double A, D, muI, DI, Cm, gL, Vre, EL, *vs, r0, p0, dv, Gint, H, p0int, *v0, vmid, p0last, Gleft, Gright, Gmid, vmidmid, Gmidmid, Gint2, HexpIntG, vright, vleft;
    double Vlb, Vth;

    // Read parameters from standard input
    scanf("%lf %lf %lf %lf %lf %lf %d %lf %lf", &muI, &DI, &Cm, &gL, &Vre, &EL, &Nv, &Vlb, &Vth);

    // Calculate the mesh values with linear spacing
    vs = (double *)malloc(Nv * sizeof(double));
    for (k = 0; k < Nv; k++) {
        vs[k] = Vlb + k * (Vth - Vlb) / (Nv - 1);
    }

    // Initialize variables
    p0 = 0;
    p0last = 0;
    p0int = 0;
    v0 = (double *)malloc(sizeof(double));
    v0[0] = 0;
    D = DI / (Cm * Cm);

    // Iterate backward through mesh
    for (k = Nv - 2; k >= 0; k--) {
        vleft=vs[k]; /* left endpoint of mesh */
        vright=vs[k+1]; /* right endpoint of mesh */
        vmid=(vleft+vright)/2; /* potential at midpoint of mesh */
        dv=vright-vleft; /* Current mesh step */


        /* Check to make sure mesh is increasing */
        if(dv<=0) {

           fprintf(stderr,  "dv negative or zero.\n");
            // Return a non-zero value to indicate an error
            return 1;}


        /** Simpson's rule for G **/

          /* Evaluate G and H at left endpoint */
          Gleft=-(-gL*(vleft-EL)+muI)/(D*Cm);

          /* Evaluate G and H at midpoint */
          Gmid=-(-gL*(vmid-EL)+muI)/(D*Cm);

          /* Evaluate G and H at right endpoint */
          Gright=-(-gL*(vright-EL)+muI)/(D*Cm);

          /* Estimate the integral of G */
          /* over entire interval */
          Gint=(1.0/6.0)*(Gleft+4*Gmid+Gright)*dv;

          /* Just evaluate H at midpoint */
          H=(vmid>Vre)*(1/D);

         /* Evaluate G halfway between left enpoint and midpoint */
         /* For use in the line below */
          vmidmid=vmid-dv/2.0;
          Gmidmid=-(-gL*(vmidmid-EL)+muI)/(D*Cm);

          /* Now estimate the integral of G from the left endpoint to the midpoint */
          Gint2=(1.0/6.0)*(Gleft+4*Gmidmid+Gmid)*(vmid-vleft);

          /* Estimate the integral of
             H*exp(Int G) over entire interval */
          HexpIntG=(1.0/6.0)*H*(1+4*exp(Gint2)+exp(Gint))*dv;

          /* update p0 */
          p0=p0*exp(Gint)+HexpIntG;

        /* Trapezoidal approximation to integral */
        p0int+=dv*((p0+p0last)/2.0);
        v0[0]+=dv*((p0*vleft+p0last*(vright))/2.0);
        p0last=p0;
    }

//    fprintf(resultFile, "r0: %lf\nv0: %lf\n", r0, v0[0]);
    r0=1/(p0int);   /* rate is 1/(integral of p0) */
    v0[0]=v0[0]*r0; /* membrane potential is (integral of p0*v)*rate */

    // Write the results to standard output
    printf("r0: %lf\nv0: %lf\n", r0, v0[0]);

    // Clean up and free memory
    free(vs);
    free(v0);

    return 0;
}

