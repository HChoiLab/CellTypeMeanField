#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    int Nv, k;
    double A, D, Re, Rp, Rs, Rv, be, bp, bs, bv, mu, Cm, gL, Vre, EL, Ee, Ei, *vs, r0, p0, dv, Gint, H, p0int, *v0, vmid, p0last, Gleft, Gright, Gmid, Gint2, HexpIntG, vmidmid, Gmidmid, vright, vleft;
    double Vlb, Vth;
    double tauL, EPrime, gLPrime;
    double Dleft, Dmid, Dright, Dmidmid;

    // Read parameters from standard input
    scanf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf", &Re, &Rp, &Rs, &Rv, &be, &bp, &bs, &bv, &mu, &Cm, &gL, &Vre, &EL, &Ee, &Ei, &Nv, &Vlb, &Vth);

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

    tauL = Cm / gL;
    EPrime = (EL +tauL*Re*(be+be*be)*Ee + tauL*Rp*(bp+bp*bp)*Ei + tauL*Rs*(bs+bs*bs)*Ei + tauL*Rv*(bv+bv*bv)*Ei) / (1 + tauL*(Re*(be+be*be) + Rp*(bp+bp*bp) + Rs*(bs+bs*bs) + Rv*(bv+bv*bv)));
    gLPrime = gL * (1 + tauL*(Re*(be+be*be) + Rp*(bp+bp*bp) + Rs*(bs+bs*bs) + Rv*(bv+bv*bv)));

    // Iterate backward through mesh
    for (k = Nv - 2; k >= 0; k--) {
        vleft = vs[k]; /* left endpoint of mesh */
        vright = vs[k+1]; /* right endpoint of mesh */
        vmid = (vleft + vright)/2; /* potential at midpoint of mesh */
        dv = vright - vleft; /* Current mesh step */

        // Variance at midpoint
        Dleft = 0.5*(Re*be*be*(vleft-Ee)*(vleft-Ee) + Rp*bp*bp*(vleft-Ei)*(vleft-Ei)+ Rs*bs*bs*(vleft-Ei)*(vleft-Ei)+ Rv*bv*bv*(vleft-Ei)*(vleft-Ei));
        Dmid = 0.5*(Re*be*be*(vmid-Ee)*(vmid-Ee) + Rp*bp*bp*(vmid-Ei)*(vmid-Ei)+ Rs*bs*bs*(vmid-Ei)*(vmid-Ei)+ Rv*bv*bv*(vmid-Ei)*(vmid-Ei));
        Dright = 0.5*(Re*be*be*(vright-Ee)*(vright-Ee) + Rp*bp*bp*(vright-Ei)*(vright-Ei)+ Rs*bs*bs*(vright-Ei)*(vright-Ei)+ Rv*bv*bv*(vright-Ei)*(vright-Ei));

        // Drift and variance terms at left, mid, and right points
        Gleft = -(-gLPrime*(vleft-EPrime) + mu) / (Dleft*Cm);
        Gmid = -(-gLPrime*(vmid-EPrime) + mu) / (Dmid*Cm);
        Gright = -(-gLPrime*(vright-EPrime) + mu) / (Dright*Cm);

        // Simpson's rule integration for Gint2 (from left to midpoint) and Gint (over the full interval)
        vmidmid = (vleft + vmid)/2;
        Dmidmid = 0.5*(Re*be*be*(vmidmid-Ee)*(vmidmid-Ee) + Rp*bp*bp*(vmidmid-Ei)*(vmidmid-Ei)+ Rs*bs*bs*(vmidmid-Ei)*(vmidmid-Ei)+ Rv*bv*bv*(vmidmid-Ei)*(vmidmid-Ei));
        Gmidmid = -(-gLPrime*(vmidmid-EPrime) + mu) / (Dmidmid*Cm);

        Gint2 = (1.0/6.0)*(Gleft + 4*Gmidmid + Gmid)*(vmid - vleft);
        Gint = (1.0/6.0)*(Gleft + 4*Gmid + Gright)*dv;

        // Simpson's rule to integrate H * exp(Int G)
        H = (vmid > Vre) * (1/Dmid);
        HexpIntG = (1.0/6.0) * H * (1 + 4 * exp(Gint2) + exp(Gint)) * dv;

        // Update p0
        p0 = p0 * exp(Gint) + HexpIntG;

        // Trapezoidal approximation to integral
        p0int += dv * ((p0 + p0last) / 2.0);
        v0[0] += dv * ((p0 * vleft + p0last * vright) / 2.0);
        p0last = p0;
    }

    r0 = 1 / (p0int);   /* rate is 1/(integral of p0) */
    v0[0] = v0[0] * r0; /* membrane potential is (integral of p0*v)*rate */

    // Write the results to standard output
    printf("r0: %lf\nv0: %lf\n", r0, v0[0]);

    // Clean up and free memory
    free(vs);
    free(v0);

    return 0;
}

