#include "marchee.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

uint * cree_DX(uint plus, uint ema) {
	uint * ret = alloc<uint>(MAX_PARAMS);
	ret[0] = plus;
	ret[1] = ema;
	return ret;
}

void nature3__dx(ema_int_t * ema_int) {
	//			-- Parametres --
	uint plus0 = ema_int->params[0];
	uint ema0  = ema_int->params[1];
	//			-- Assertions --
	ASSERT(min_param[DX][0] <= plus0 && plus0 <= max_param[DX][0]);
	ASSERT(min_param[DX][1] <= ema0  && ema0  <= max_param[DX][1]);
	//	-- Transformation des Parametres --
	//		-- Calcule de la Nature --
	_outil_dx (ema_int->brute, ema_int->ema, plus0);
	_outil_ema(ema_int->brute, ema_int->brute, ema0);
};