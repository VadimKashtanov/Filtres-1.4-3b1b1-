#include "marchee.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

uint * cree_DXDX(uint plus0, uint ema0, uint plus1, uint ema1) {
	uint * ret = alloc<uint>(MAX_PARAMS);
	ret[0] = plus0;
	ret[1] =  ema0;
	ret[2] = plus1;
	ret[3] =  ema1;
	return ret;
}

void nature4__dxdx(ema_int_t * ema_int) {
	//			-- Parametres --
	uint plus0 = ema_int->params[0];
	uint ema0  = ema_int->params[1];
	uint plus1 = ema_int->params[2];
	uint ema1  = ema_int->params[3];
	//			-- Assertions --
	ASSERT(min_param[DXDX][0] <= plus0 && plus0 <= max_param[DXDX][0]);
	ASSERT(min_param[DXDX][1] <= ema0  && ema0  <= max_param[DXDX][1]);
	ASSERT(min_param[DXDX][2] <= plus1 && plus1 <= max_param[DXDX][2]);
	ASSERT(min_param[DXDX][3] <= ema1  && ema1  <= max_param[DXDX][3]);
	//	-- Transformation des Parametres --
	//		-- Calcule de la Nature --
	_outil_dx (ema_int->brute, ema_int->ema, plus0);
	_outil_ema(ema_int->brute, ema_int->brute, ema0);
	_outil_dx (ema_int->brute, ema_int->brute, plus1);
	_outil_ema(ema_int->brute, ema_int->brute, ema1);
};