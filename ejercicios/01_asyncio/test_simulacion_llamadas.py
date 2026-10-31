import asyncio
import time

import pytest

from simulacion_llamadas import ejecutar_con_semaforo, ejecutar_con_timeout_total

pytestmark = pytest.mark.asyncio


def llamada_simulada(segundos: float):
    # Llamada falsa con latencia corta para que las pruebas corran rápido
    async def llamada(prompt: str) -> dict:
        await asyncio.sleep(segundos)
        return {"respuesta": f"ok en {segundos} s"}

    return llamada


async def test_las_llamadas_corren_en_paralelo():
    llamadas = {"a": llamada_simulada(0.1), "b": llamada_simulada(0.2), "c": llamada_simulada(0.3)}

    inicio = time.perf_counter()
    resultados = await ejecutar_con_timeout_total(llamadas, "hola", timeout=1.0)
    duracion = time.perf_counter() - inicio

    assert all(r is not None for r in resultados.values())
    # En serie tardarían 0.6 s; en paralelo, lo que tarda la más lenta (0.3 s)
    assert duracion < 0.45


async def test_timeout_total_cancela_la_lenta_y_conserva_las_demas():
    llamadas = {"rapida": llamada_simulada(0.05), "lenta": llamada_simulada(1.0)}

    resultados = await ejecutar_con_timeout_total(llamadas, "hola", timeout=0.2)

    assert resultados["rapida"] == {"respuesta": "ok en 0.05 s"}
    assert resultados["lenta"] is None


async def test_semaforo_limita_la_concurrencia():
    activas = 0
    maximo = 0

    async def llamada_contada(prompt: str) -> dict:
        nonlocal activas, maximo
        activas += 1
        maximo = max(maximo, activas)
        await asyncio.sleep(0.05)
        activas -= 1
        return {"respuesta": "ok"}

    llamadas = [(f"#{i}", llamada_contada) for i in range(10)]
    resultados = await ejecutar_con_semaforo(llamadas, "hola", max_concurrencia=2, timeout_por_llamada=1.0)

    assert len(resultados) == 10
    assert maximo == 2


async def test_timeout_por_llamada_no_afecta_a_las_demas():
    llamadas = [("lenta", llamada_simulada(1.0)), ("rapida", llamada_simulada(0.05))]

    resultados = await ejecutar_con_semaforo(llamadas, "hola", max_concurrencia=2, timeout_por_llamada=0.2)

    assert isinstance(resultados[0], TimeoutError)
    assert resultados[1] == {"respuesta": "ok en 0.05 s"}


async def test_el_tiempo_en_cola_no_cuenta_para_el_timeout():
    # Con un solo lugar, la segunda espera 0.15 s en cola y luego corre 0.15 s:
    # termina a los 0.30 s, pero su timeout de 0.2 s empieza al conseguir lugar
    llamadas = [("primera", llamada_simulada(0.15)), ("segunda", llamada_simulada(0.15))]

    resultados = await ejecutar_con_semaforo(llamadas, "hola", max_concurrencia=1, timeout_por_llamada=0.2)

    assert not any(isinstance(r, Exception) for r in resultados)
