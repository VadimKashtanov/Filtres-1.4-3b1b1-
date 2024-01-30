#include "mdl.cuh"

static uint couche_aleatoire(Mdl_t * mdl) {
	uint a = rand() % mdl->total_POIDS;
	FOR(0, i, C) {
		if (a < mdl->inst_POIDS[i]) {
			return i;
		} else {
			a -= mdl->inst_POIDS[i];
		}
	}
	return C-1;
}

//	===================================================

static void perturber_filtre(Mdl_t * mdl) {
	uint f = rand() % (BLOQUES*F_PAR_BLOQUES);
	//
	float r[N];
	r[0] = rnd();
	FOR(1, i, N) r[i] = r[i-1] + rnd()-.5;
	//
	float coef = 0.95;
	FOR(0, i, N) mdl->p[0][f*N + i] = mdl->p[0][f*N + i]*coef + (1-coef)*r[i];
};

void perturber_filtres(Mdl_t * mdl, uint L) {
	FOR(0, i, L) perturber_filtre(mdl);
};

//	===================================================

static void perturber_fois_zero  (Mdl_t * mdl, uint c) {
	if (mdl->insts[c] == DOT1D) {
		uint X=mdl->Y[c-1], Y=mdl->Y[c];
		mdl->p[c][(X+1)*(rand()%Y) + (rand()%X)] /= (float)(1+rand()%3);
	}
};

static void perturber_echanger   (Mdl_t * mdl, uint c) {
	if (mdl->insts[c] == DOT1D) {
		uint X=mdl->Y[c-1], Y=mdl->Y[c];
		uint p0 = (X+1)*(rand()%Y) + (rand()%X);
		uint p1 = (X+1)*(rand()%Y) + (rand()%X);
		float vp0 = mdl->p[c][p0], vp1 = mdl->p[c][p1];
		mdl->p[c][p0] = vp1;
		mdl->p[c][p1] = vp0;
	}
};

static void perturber_egale_rnd  (Mdl_t * mdl, uint c) {
	if (mdl->insts[c] == DOT1D) {
		uint X=mdl->Y[c-1], Y=mdl->Y[c];
		mdl->p[c][(X+1)*(rand()%Y) + (rand()%X)] += 0.2 * 2*(rnd()-0.5);
	}
};

//	======================================

typedef void (*perturber_f)(Mdl_t * mdl, uint c);

static perturber_f arr_perturber_f[3] = {
	perturber_fois_zero,
	perturber_echanger,
	perturber_egale_rnd
};

void perturber(Mdl_t * mdl, uint L) {
	mdl_gpu_vers_cpu(mdl);
	FOR(0, i, L) {
		uint c_alea = couche_aleatoire(mdl);
		if (c_alea != 0) arr_perturber_f[rand() % 3](mdl, c_alea);
		else                        perturber_filtre(mdl);
	}
	mdl_cpu_vers_gpu(mdl);
	//
	if (NORMER_LES_FILTRES) mdl_normer_les_filtres(mdl);
	if (BORNER_LES_FILTRES) mdl_borner_les_filtres(mdl);
};