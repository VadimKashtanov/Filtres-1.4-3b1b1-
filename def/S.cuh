#pragma once

#include "marchee.cuh"

#define SCORE_Y_COEF_BRUIT 0.30

#define POW_individuel 2
#define POW_somme      1//1
#define POW_coef       1.0

#define sng(x)	((x>=0) ? 1.0 : -1.0)

#define MODE_reduction_ERREUR 0
#define MODE_reduction_PERTES 1

#define MODE_SCORE MODE_reduction_ERREUR

#if (MODE_SCORE == MODE_reduction_ERREUR)
	#define __SCORE(y,p1,p0)  (powf(fabs(100*(p1/p0 - 1)),POW_coef) * powf((y) - sng(p1/p0 - 1), POW_individuel)/POW_individuel)
	#define __dSCORE(y,p1,p0) (powf(fabs(100*(p1/p0 - 1)),POW_coef) * powf(y - sng(p1/p0 - 1), POW_individuel-1))
#elif (MODE_SCORE == MODE_reduction_PERTES)
#define __MUL 100
	#define __SCORE(y,p1,p0)  (powf( - y*__MUL*(p1/p0-1), POW_individuel)/POW_individuel)
	#define __dSCORE(y,p1,p0) (-__MUL*(p1/p0-1)*powf(-y*__MUL*(p1/p0-1), POW_individuel-1))
#endif


//	----

static float SCORE(float y, float p1, float p0) {
	return __SCORE(y,p1,p0);
};

static float APRES_SCORE(float somme) {
	return powf(somme, POW_somme) / POW_somme;
};

static float dAPRES_SCORE(float somme) {
	return powf(somme, POW_somme - 1);
};

static float dSCORE(float y, float p1, float p0) {
	return __dSCORE(y,p1,p0);
};

//	----

static __device__ float cuda_SCORE(float y, float p1, float p0) {
	return __SCORE(y,p1,p0);
};

static __device__ float cuda_dSCORE(float y, float p1, float p0) {
	return __dSCORE(y,p1,p0);
};

//	S(x) --- Score ---

float  intel_somme_score(float * y, uint depart, uint T);
float nvidia_somme_score(float * y, uint depart, uint T);

float  intel_score_finale(float somme, uint T);
float nvidia_score_finale(float somme, uint T);

//	dx

float d_intel_score_finale(float somme, uint T);
float d_nvidia_score_finale(float somme, uint T);

void  d_intel_somme_score(float d_somme, float * y, float * dy, uint depart, uint T);
void d_nvidia_somme_score(float d_somme, float * y, float * dy, uint depart, uint T);

//	%% Prediction

float* intel_prediction(float * y, uint depart, uint T);
float* nvidia_prediction(float * y, uint depart, uint T);