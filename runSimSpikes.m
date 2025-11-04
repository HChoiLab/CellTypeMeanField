% This program runs the network simulation for the given parameter file.
% Spike times are returned.
%
% Before running this, you must compile the function LIF4popfr by entering:
% mex LIF4popfr.c
%
% This code is based on the code by Robert Rosenbaum (Rosenbaum and Doiron, Physical Review X, 4:021039 2007)
% Adapted by Soon Ho Kim to implement inhibitory cell types PV/SST/VIP (Kim and Choi, bioRxiv 2025.03.13.643046)

% Clear all variables

function runSimSpikes(parname, Nn, seed)

% Time the network simulation
tic

run(strcat('sim_parameter_files/',parname,'.m'));
folderName = parname
N = Nn

% Create a folder with the name parname in the results folder
results_path = fullfile('sim_results', parname);
if ~exist(results_path, 'dir')
    mkdir(results_path);
end

% Copy file with file name parname+'.m' to folder sim_results/parname
source_file = fullfile('sim_parameter_files', [parname, '.m']);
[~, basename, ~] = fileparts(parname);
destination_file = fullfile(results_path, [basename, '.m']);
copyfile(source_file, destination_file);

rng(seed);

% Simulation time and bin size (in msec)
Nt=round(T/dt);

% Number of neurons in network

% Scaling factors for connection probability (k) and connection strength
% (j)
% To Exc
kee=kbar_ee*knorm;
kep=kbar_ep*knorm;
kes=kbar_es*knorm;
kev=kbar_ev*knorm;

% To pv
kpe=kbar_pe*knorm;
kpp=kbar_pp*knorm;
kps=kbar_ps*knorm;
kpv=kbar_pv*knorm;

% To SST
kse=kbar_se*knorm;
ksp=kbar_sp*knorm;
kss=kbar_ss*knorm;
ksv=kbar_sv*knorm;

% To VIP
kve=kbar_ve*knorm;
kvp=kbar_vp*knorm;
kvs=kbar_vs*knorm;
kvv=kbar_vv*knorm;

% Number of exc. and inh. neurons
qe = 1 - qp - qs - qv;
Np=round(N*qp);
Ns=round(N*qs);
Nv=round(N*qv);
% N=Ne+Np+Ns+Nv;
Ne = N - Np - Ns - Nv
disp(sprintf('\nNumber of Exc: %d, PV: %d, SST: %d, VIP: %d\n',Ne, Np, Ns, Nv));
disp(sprintf('\nNumber of Neurons: %d\n',N));

% Threshold, reset, lower boundary, membrane time constant of LIFs
gL=zeros(N,1);
% gL(1:Ne)=1/taume;
% gL((Ne+1):(Ne+Np))=1/taump;
% gL((Ne+Np+1):(Ne+Np+Ns))=1/taums;
for j=1:Ne
gL(j)=1/taume;
end
for j=1:Np
gL(j+Ne)=1/taump;
end
for j=1:Ns
gL(j+Ne+Np)=1/taums;
end
for j=1:Nv
gL(j+Ne+Np+Ns)=1/taumv;
end

% For balanced state to exist this vector should be decreasing
% disp(sprintf('\nThis list should be decreasing for\n  a balanced state to exist: %.2f %.2f %.2f\n',jExte/jExti,jei/jii,jee/jie));

% and these values should be >1
% disp(sprintf('\nAlso, this number should be greater than 1: %.2f\n',jii/jee));


% Wrapped Gaussian with mean mmu and std ssigma.  
% Wraps a Gaussian around the interval [0,1].
% xx is the argument and should lie in [0,1].
% KK is the number of times to wrap the Gaussian.  Should be large enough
% so that the unwrapped gaussian is nearly zero at x=KK.
WrappedGauss=@(xx,mmu,ssigma,KK)(1/(ssigma*sqrt(2*pi))*sum(exp(-(xx-mmu+(-KK:KK)).^2/(2*ssigma^2))));
kk=50;


% Scaling factor.  Making this small reduces sycnrhony at finite N.
%  K=round(N/100);
K=N;

% Width of inputs on [0 N] instead of [0 1]
betaee=sigmaee*(Ne);
betaep=sigmaep*(Ne);
betaes=sigmaes*(Ne);
betaev=sigmaev*(Ne);
betape=sigmape*(Np);
betapp=sigmapp*(Np);
betaps=sigmaps*(Np);
betapv=sigmapv*(Np);
betase=sigmase*(Ns);
betasp=sigmasp*(Ns);
betass=sigmass*(Ns);
betasv=sigmasv*(Ns);
betave=sigmave*(Nv);
betavp=sigmavp*(Nv);
betavs=sigmavs*(Nv);
betavv=sigmavv*(Nv);

betaExte=sigmaExte*(Ne);
betaExtp=sigmaExtp*(Np);
betaExts=sigmaExts*(Ns);
betaExtv=sigmaExtv*(Nv);

% Number of inputs to each neuron
Kee=round(kee*Ne);
Kep=round(kep*Ne);
Kes=round(kes*Ne);
Kev=round(kev*Ne);
Kpe=round(kpe*Np);
Kpp=round(kpp*Np);
Kps=round(kps*Np);
Kpv=round(kpv*Np);
Kse=round(kse*Ns);
Ksp=round(ksp*Ns);
Kss=round(kss*Ns);
Ksv=round(ksv*Ns);
Kve=round(kve*Nv);
Kvp=round(kvp*Nv);
Kvs=round(kvs*Nv);
Kvv=round(kvv*Nv);

% Synaptic strengths
Jee=(jee/sqrt(K));
Jep=-(jep/sqrt(K));
Jes=-(jes/sqrt(K));
Jev=-(jev/sqrt(K));
Jpe=(jpe/sqrt(K));
Jpp=-(jpp/sqrt(K));
Jps=-(jps/sqrt(K));
Jpv=-(jpv/sqrt(K));
Jse=(jse/sqrt(K));
Jsp=-(jsp/sqrt(K));
Jss=-(jss/sqrt(K));
Jsv=-(jsv/sqrt(K));
Jve=(jve/sqrt(K));
Jvp=-(jvp/sqrt(K));
Jvs=-(jvs/sqrt(K));
Jvv=-(jvv/sqrt(K));

% Strength of external 
JExte=jExte*sqrt(K);
JExtp=jExtp*sqrt(K);
JExts=jExts*sqrt(K);
JExtv=jExtv*sqrt(K);


% External input to excitatory and inhibitory cells
I0e=zeros(Ne,1);
I0p=zeros(Np,1);
I0s=zeros(Ns,1);
I0v=zeros(Nv,1);
for j=1:Ne
I0e(j)=(JExte*(1-pExte)+JExte*pExte*WrappedGauss(j/Ne,1/2,sigmaExte,kk));
end

for j=1:Np
I0p(j)=(JExtp*(1-pExtp)+JExtp*pExtp*WrappedGauss(j/Np,1/2,sigmaExtp,kk));
end

for j=1:Ns
I0s(j)=(JExts*(1-pExts)+JExts*pExts*WrappedGauss(j/Ns,1/2,sigmaExts,kk));
end

for j=1:Nv
I0v(j)=(JExtv*(1-pExtv)+JExtv*pExtv*WrappedGauss(j/Nv,1/2,sigmaExtv,kk));
end

% External input to entire network
Iapp=zeros(N,1);
Iapp(1:Ne)=I0e;
Iapp((Ne+1):(Ne+Np))=I0p;
Iapp((Ne+Np+1):(Ne+Np+Ns))=I0s;
Iapp((Ne+Np+Ns+1):(Ne+Np+Ns+Nv))=I0v;

% We don''t need these anymore
clear I0e I0p I0s I0v;

% Generate n numbers from a discretized wrapped Gaussian distribution
% with center at mu, width sigma, minimum min, maximum max.
% This basically generates n normally distributed variables with mean mu
% and standard deviation sigma. Then rounds the values and properly mods
% them so that they are integers between min and max.
CircRandN = @(mu, sigma, min, max, n) (mod(round(sigma * randn(n, 1) + mu) - min, max - min + 1) + min);

% Store connections.
% W is a vector containing postsynaptic cell indices, sorted by the index of the presynaptic cell.
% I0[j] denotes the index into W containing the first postsynaptic cell index of presynaptic cell j.
% Thus, for example, if I0[3] == 100 and I0[4] == 150 then the postsynaptic targets of cell 3 are stored in
% W[100] through W[149]. The postsynaptic targets of cell 4 are stored in W[150] through W[I0[5]], etc.
% The implementation below actually yields a fixed number of postsynaptic
% targets for each presynaptic cell. This gives a bit less variability
% than an actual Erdos-Renyi network, but the difference is small and this
% is easier to code.
Ke = Kee + Kpe + Kse + Kve; % Number of excitatory and inhibitory connections per cell
Kp = Kep + Kpp + Ksp + Kvp;
Ks = Kes + Kps + Kss + Kvs;
Kv = Kev + Kpv + Ksv + Kvv;
W = zeros(Ke * Ne + Kp * Np + Ks * Ns + Kv * Nv, 1);
I0 = zeros(N + 1, 1);
disp(sprintf('number of e, p, s, v projections per neuron: %d %d %d %d', Ke, Kp, Ks, Kv));
disp(sprintf('\nNumber of Exc: %d, PV: %d, SST: %d, VIP: %d\n', Ne, Np, Ns, Nv));
disp(sprintf('Kee = %d, Kpe = %d, Kse = %d, Kve = %d', Kee, Kpe, Kse, Kve));
disp(sprintf('Kep = %d, Kpp = %d, Ksp = %d, Kvp = %d', Kep, Kpp, Ksp, Kvp));
disp(sprintf('Kes = %d, Kps = %d, Kss = %d, Kvs = %d', Kes, Kps, Kss, Kvs));
disp(sprintf('Kev = %d, Kpv = %d, Ksv = %d, Kvv = %d', Kev, Kpv, Ksv, Kvv));
disp(sprintf('Total number of connections: %e', Ke * Ne + Kp * Np + Ks * Ns + Kv * Nv));

for j = 1:Ne
    % E pre, E post
    W((1 + (j - 1) * Ke):(Kee + (j - 1) * Ke)) = CircRandN(j, betaee, 1, Ne, Kee);
    if any(W((1 + (j - 1) * Ke):(Kee + (j - 1) * Ke)) == 0)
        error('Negative value found in W for E pre, E post at j = %d', j);
    end

    % E pre, PV post
    W((Kee + 1 + (j - 1) * Ke):(Kee + Kpe + (j - 1) * Ke)) = CircRandN(j * Np / Ne + Ne, betape, Ne + 1, Np + Ne, Kpe);
    if any(W((Kee + 1 + (j - 1) * Ke):(Kee + Kpe + (j - 1) * Ke)) == 0)
        error('Negative value found in W for E pre, PV post at j = %d', j);
    end

    % E pre, SST post
    W((Kee + Kpe + 1 + (j - 1) * Ke):(Kee + Kpe + Kse + (j - 1) * Ke)) = CircRandN(j * Ns / Ne + Ne + Np, betase, Ne + Np + 1, Np + Ns + Ne, Kse);
    if any(W((Kee + Kpe + 1 + (j - 1) * Ke):(Kee + Kpe + Kse + (j - 1) * Ke)) == 0)
        error('Negative value found in W for E pre, SST post at j = %d', j);
    end

    % E pre, VIP post
    W((Kee + Kpe + Kse + 1 + (j - 1) * Ke):(Kee + Kpe + Kse + Kve + (j - 1) * Ke)) = CircRandN(j * Nv / Ne + Ne + Np + Ns, betave, Ne + Np + Ns + 1, Np + Ns + Ne + Nv, Kve);
    if any(W((Kee + Kpe + Kse + 1 + (j - 1) * Ke):(Kee + Kpe + Kse + Kve + (j - 1) * Ke)) == 0)
        error('Negative value found in W for E pre, VIP post at j = %d', j);
    end

    % disp(sprintf('Made E connections from %d to %d', 1 + (j - 1) * Ke, Kee + Kpe + Kse + Kve + (j - 1) * Ke));
    I0(j) = (1 + (j - 1) * Ke);
end
disp('Made connections from E');

for j = 1:Np
    % PV pre, E post
    W((Ne * Ke + 1 + (j - 1) * Kp):(Ne * Ke + Kep + (j - 1) * Kp)) = CircRandN(j * Ne / Np, betaep, 1, Ne, Kep);
    if any(W((Ne * Ke + 1 + (j - 1) * Kp):(Ne * Ke + Kep + (j - 1) * Kp)) == 0)
        error('Negative value found in W for PV pre, E post at j = %d', j);
    end

    % PV pre, PV post
    W((Ne * Ke + Kep + 1 + (j - 1) * Kp):(Ne * Ke + Kep + Kpp + (j - 1) * Kp)) = CircRandN(j + Ne, betapp, Ne + 1, Ne + Np, Kpp);
    if any(W((Ne * Ke + Kep + 1 + (j - 1) * Kp):(Ne * Ke + Kep + Kpp + (j - 1) * Kp)) == 0)
        error('Negative value found in W for PV pre, PV post at j = %d', j);
    end

    % PV pre, SST post
    W((Ne * Ke + Kep + Kpp + 1 + (j - 1) * Kp):(Ne * Ke + Kep + Kpp + Ksp + (j - 1) * Kp)) = CircRandN(j * Ns / Np + Ne + Np, betasp, Ne + Np + 1, Ne + Np + Ns, Ksp);
    if any(W((Ne * Ke + Kep + Kpp + 1 + (j - 1) * Kp):(Ne * Ke + Kep + Kpp + Ksp + (j - 1) * Kp)) == 0)
        error('Negative value found in W for PV pre, SST post at j = %d', j);
    end

    % PV pre, VIP post
    W((Ne * Ke + Kep + Kpp + Ksp + 1 + (j - 1) * Kp):(Ne * Ke + Kep + Kpp + Ksp + Kvp + (j - 1) * Kp)) = CircRandN(j * Nv / Np + Ne + Np + Ns, betavp, Ne + Np + Ns + 1, Ne + Np + Ns + Nv, Kvp);
    if any(W((Ne * Ke + Kep + Kpp + Ksp + 1 + (j - 1) * Kp):(Ne * Ke + Kep + Kpp + Ksp + Kvp + (j - 1) * Kp)) == 0)
        error('Negative value found in W for PV pre, VIP post at j = %d', j);
    end

    % disp(sprintf('Made PV connections from %d to %d', Ne * Ke + 1 + (j - 1) * Kp, Ne * Ke + Kep + Kpp + Ksp + Kvp + (j - 1) * Kp));
    I0(j + Ne) = (Ne * Ke + 1 + (j - 1) * Kp);
end
disp('Made connections from PV');

for j = 1:Ns
    % SST pre, E post
    W((Ne * Ke + Np * Kp + 1 + (j - 1) * Ks):(Ne * Ke + Np * Kp + Kes + (j - 1) * Ks)) = CircRandN(j * Ne / Ns, betaes, 1, Ne, Kes);
    if any(W((Ne * Ke + Np * Kp + 1 + (j - 1) * Ks):(Ne * Ke + Np * Kp + Kes + (j - 1) * Ks)) == 0)
        error('Negative value found in W for SST pre, E post at j = %d', j);
    end

    % SST pre, PV post
    W((Ne * Ke + Np * Kp + Kes + 1 + (j - 1) * Ks):(Ne * Ke + Np * Kp + Kes + Kps + (j - 1) * Ks)) = CircRandN(j * Np / Ns + Ne, betaps, Ne + 1, Ne + Np, Kps);
    if any(W((Ne * Ke + Np * Kp + Kes + 1 + (j - 1) * Ks):(Ne * Ke + Np * Kp + Kes + Kps + (j - 1) * Ks)) == 0)
        error('Negative value found in W for SST pre, PV post at j = %d', j);
    end

    % SST pre, SST post
    W((Ne * Ke + Np * Kp + Kes + Kps + 1 + (j - 1) * Ks):(Ne * Ke + Np * Kp + Kes + Kps + Kss + (j - 1) * Ks)) = CircRandN(j + Ne + Np, betass, Ne + Np + 1, Ne + Np + Ns, Kss);
    if any(W((Ne * Ke + Np * Kp + Kes + Kps + 1 + (j - 1) * Ks):(Ne * Ke + Np * Kp + Kes + Kps + Kss + (j - 1) * Ks)) == 0)
        error('Negative value found in W for SST pre, SST post at j = %d', j);
    end

    % SST pre, VIP post
    W((Ne * Ke + Np * Kp + Kes + Kps + Kss + 1 + (j - 1) * Ks):(Ne * Ke + Np * Kp + Kes + Kps + Kss + Kvs + (j - 1) * Ks)) = CircRandN(j * Nv / Ns + Ne + Np + Ns, betavs, Ne + Np + Ns + 1, Ne + Np + Ns + Nv, Kvs);
    if any(W((Ne * Ke + Np * Kp + Kes + Kps + Kss + 1 + (j - 1) * Ks):(Ne * Ke + Np * Kp + Kes + Kps + Kss + Kvs + (j - 1) * Ks)) == 0)
        error('Negative value found in W for SST pre, VIP post at j = %d', j);
    end

    % disp(sprintf('Made SST connections from %d to %d', Ne * Ke + Np * Kp + 1 + (j - 1) * Ks,Ne * Ke + Np * Kp + Kes + Kps + Kss + Kvs + (j - 1) * Ks));
    I0(j + Ne + Np) = (Ne * Ke + Np * Kp + 1 + (j - 1) * Ks);
end
disp('Made connections from SST');

for j = 1:Nv
    % VIP pre, E post
    W((Ne * Ke + Np * Kp + Ns * Ks + 1 + (j - 1) * Kv):(Ne * Ke + Np * Kp + Ns * Ks + Kev + (j - 1) * Kv)) = CircRandN(j * Ne / Nv, betaev, 1, Ne, Kev);
    if any(W((Ne * Ke + Np * Kp + Ns * Ks + 1 + (j - 1) * Kv):(Ne * Ke + Np * Kp + Ns * Ks + Kev + (j - 1) * Kv)) == 0)
        error('Negative value found in W for VIP pre, E post at j = %d', j);
    end

    % VIP pre, PV post
    W((Ne * Ke + Np * Kp + Ns * Ks + Kev + 1 + (j - 1) * Kv):(Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + (j - 1) * Kv)) = CircRandN(j * Np / Nv + Ne, betapv, Ne + 1, Ne + Np, Kpv);
    if any(W((Ne * Ke + Np * Kp + Ns * Ks + Kev + 1 + (j - 1) * Kv):(Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + (j - 1) * Kv)) == 0)
        error('Negative value found in W for VIP pre, PV post at j = %d', j);
    end

    % VIP pre, SST post
    W((Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + 1 + (j - 1) * Kv):(Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + Ksv + (j - 1) * Kv)) = CircRandN(j * Ns / Nv + Ne + Np, betasv, Ne + Np + 1, Ne + Np + Ns, Ksv);
    if any(W((Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + 1 + (j - 1) * Kv):(Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + Ksv + (j - 1) * Kv)) == 0)
        error('Negative value found in W for VIP pre, SST post at j = %d', j);
    end

    % VIP pre, VIP post
    W((Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + Ksv + 1 + (j - 1) * Kv):(Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + Ksv + Kvv + (j - 1) * Kv)) = CircRandN(j + Ne + Np + Ns, betavv, Ne + Np + Ns + 1, Ne + Np + Ns + Nv, Kvv);
    if any(W((Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + Ksv + 1 + (j - 1) * Kv):(Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + Ksv + Kvv + (j - 1) * Kv)) == 0)
        error('Negative value found in W for VIP pre, VIP post at j = %d', j);
    end

     % disp(sprintf('Made VIP connections from %d to %d', Ne * Ke + Np * Kp + Ns * Ks + 1 + (j - 1) * Kv,(Ne * Ke + Np * Kp + Ns * Ks + Kev + Kpv + Ksv + Kvv + (j - 1) * Kv)));
    I0(j + Ne + Np + Ns) = (Ne * Ke + Np * Kp + Ns * Ks + 1 + (j - 1) * Kv);
end
disp('Made connections from VIP');

if any(W == 0)
disp('Found zero in W');
% Print first and last indices of W that are zero
disp(sprintf('First zero in W: %d', find(W == 0, 1, 'first')));
disp(sprintf('Last zero in W: %d', find(W == 0, 1, 'last')));
% Print number of zeros in W
disp(sprintf('Number of zeros in W: %d', sum(W == 0)));
error('Negative value found in W');
end

I0(N + 1) = Ne * Ke + Np * Kp + Ns * Ks + Nv * Kv + 1;
  
% Random initial membrane potentials
V0=(V0max-V0min).*rand(N,1)+V0min;

% Maximum number of spikes.
% Simulation will terminate with a warning if this is exceeded

% Run network simulation
maxns=N*T*.5;
s=LIF4pop(W,I0,Np,Ns,Nv,Iapp,Jee,Jep,Jes,Jev,Jpe,Jpp,Jps,Jpv,Jse,Jsp,Jss,Jsv,Jve,Jvp,Jvs,Jvv,gL,Vth,Vre,Vlb,V0,T,dt,maxns);
disp(sprintf('\nSimulation Complete.\n'));
save(sprintf('sim_results/%s/spikes_%d_seed_%d.mat', parname, N, seed),'s');
                  
% The simulation took t0 seconds to run


t0=toc;
disp(sprintf('\nThe simulation took %.1f seconds to run\n',t0));

J=find(s(2,:)>0);
s=s(:,J);
clear J; % Close all open figures.
close all;

dX=.005;
X=dX:dX:1;
nX=numel(X);

% Create a figure without displaying it
%fig = figure('Visible', 'off');

% Plot exc and inh rasters
%make first subplot two times as tall as the second and third
figure
subplot(4,1,1)

plot(s(1,s(2,:)<=Ne),s(2,s(2,:)<=Ne)/Ne,'.','MarkerSize',.0005,'MarkerEdgeColor','black')
ylabel('E cell location')
subplot(4,1,2)
% plot(s(1,Ne+Np>s(2,:)>Ne),(s(2,Ne+Np>s(2,:)>Ne)-Ne)/Np,'.','MarkerSize',.001)
%make marker color blue
plot(s(1,s(2,:)>Ne & s(2,:)<=Ne+Np),(s(2,s(2,:)>Ne & s(2,:)<=Ne+Np)-Ne)/Np,'.','MarkerSize',.0005,'MarkerEdgeColor','b')
ylabel('PV cell location')
subplot(4,1,3)
%make marker color orange
plot(s(1,s(2,:)>Ne+Np & s(2,:)<=Ne+Np+Ns),(s(2,s(2,:)>Ne+Np & s(2,:)<=Ne+Np+Ns)-Ne-Np)/Ns,'.','MarkerSize',.0005, 'MarkerEdgeColor',[1 0.5 0])
ylabel('SST cell location')
subplot(4,1,4)
%make marker color orange
plot(s(1,s(2,:)>Ne+Np+Ns),(s(2,s(2,:)>Ne+Np+Ns)-Ne-Np-Ns)/Nv,'.','MarkerSize',.0005, 'MarkerEdgeColor','g')
ylabel('VIP cell location')
xlabel('time (msec)')

t_min = 0;
t_max = 20;
for i = 1:4
    subplot(4,1,i)
    xlim([t_min, t_max]);
end

% Save raster plot as svg file
print(sprintf('sim_results/%s/raster_s4_%d_seed_%d.png', parname, N, seed), '-dpng', '-r300');

%another raster from 200-220
figure
subplot(4,1,1)

plot(s(1,s(2,:)<=Ne),s(2,s(2,:)<=Ne)/Ne,'.','MarkerSize',.0005,'MarkerEdgeColor','black')
ylabel('E cell location')
subplot(4,1,2)
% plot(s(1,Ne+Np>s(2,:)>Ne),(s(2,Ne+Np>s(2,:)>Ne)-Ne)/Np,'.','MarkerSize',.001)
%make marker color blue
plot(s(1,s(2,:)>Ne & s(2,:)<=Ne+Np),(s(2,s(2,:)>Ne & s(2,:)<=Ne+Np)-Ne)/Np,'.','MarkerSize',.0005,'MarkerEdgeColor','b')
ylabel('PV cell location')
subplot(4,1,3)
%make marker color orange
plot(s(1,s(2,:)>Ne+Np & s(2,:)<=Ne+Np+Ns),(s(2,s(2,:)>Ne+Np & s(2,:)<=Ne+Np+Ns)-Ne-Np)/Ns,'.','MarkerSize',.0005, 'MarkerEdgeColor',[1 0.5 0])
ylabel('SST cell location')
subplot(4,1,4)
%make marker color orange
plot(s(1,s(2,:)>Ne+Np+Ns),(s(2,s(2,:)>Ne+Np+Ns)-Ne-Np-Ns)/Nv,'.','MarkerSize',.0005, 'MarkerEdgeColor','g')
ylabel('VIP cell location')
xlabel('time (msec)')

t_min = 200;
t_max = 210;
for i = 1:4
    subplot(4,1,i)
    xlim([t_min, t_max]);
end

% Save raster plot as svg file
print(sprintf('sim_results/%s/raster_s4200_%d_seed_%d.png', parname, N, seed), '-dpng', '-r300');

% Discretize space to calculate rates
dX=.005;
X=dX:dX:1;
nX=numel(X);

%fig = figure('Visible', 'off');
% Compute firing rates over entire simualtion with discretized space
Erates=hist(s(2,s(2,:)<=Ne & s(1,:)>=Tburn)/Ne,X-dX/2);
PVrates=hist((s(2, s(2,:)>Ne & s(2,:)<Ne+Np+1 & s(1,:)>=Tburn)-Ne)/Np,X-dX/2);
SSTrates=hist((s(2,s(2,:)>Ne+Np & s(2,:)<Ne+Np+Ns+1 & s(1,:)>=Tburn)-Ne-Np)/Ns,X-dX/2);
VIPrates=hist((s(2,s(2,:)>Ne+Np+Ns & s(2,:)<Ne+Np+Ns+Nv+1 & s(1,:)>=Tburn)-Ne-Np-Ns)/Nv,X-dX/2);

% Scale rates to Hz
Erates=1000*Erates/(Ne*dX*(T-Tburn));
PVrates=1000*PVrates/(Np*dX*(T-Tburn));
SSTrates=1000*SSTrates/(Ns*dX*(T-Tburn));
VIPrates=1000*VIPrates/(Nv*dX*(T-Tburn));

% Plot rates
figure
subplot(4,1,1)
plot(X,Erates)
ylabel('E rate (Hz)')
subplot(4,1,2)
plot(X,PVrates)
ylabel('PV rate (Hz)')
subplot(4,1,3)
plot(X,SSTrates)
ylabel('SST rate (Hz)')
subplot(4,1,4)
plot(X,VIPrates)
ylabel('VIP rate (Hz)')
xlabel('cell location')

save(sprintf('sim_results/%s/erates_%d_seed_%d.mat', parname, N, seed),'Erates');
save(sprintf('sim_results/%s/pvrates_%d_seed_%d.mat', parname, N, seed),'PVrates');
save(sprintf('sim_results/%s/sstrates_%d_seed_%d.mat', parname, N, seed),'SSTrates');
save(sprintf('sim_results/%s/viprates_%d_seed_%d.mat', parname, N, seed),'VIPrates');
 
print(sprintf('sim_results/%s/frates_s4_%d_seed_%d.png', parname, N, seed), '-dpng', '-r300');


end
