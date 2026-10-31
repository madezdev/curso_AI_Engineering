import asyncio
import time
from collections.abc import Awaitable, Callable

TIMEOUT_TOTAL = 2.0
TIMEOUT_POR_LLAMADA = 2.0
MAX_CONCURRENCIA = 2
INICIO = time.perf_counter()

Llamada = Callable[[str], Awaitable[dict]]


def log(mensaje: str) -> None:
    # Antepone los segundos transcurridos desde el arranque para ver la concurrencia
    print(f"[{time.perf_counter() - INICIO:5.2f}s] {mensaje}")


async def gpt_4_call(prompt: str) -> dict:
    # Simula ~2.5 s de latencia de red hacia la API de OpenAI (supera el timeout)
    await asyncio.sleep(2.5)
    return {"modelo": "gpt-4", "respuesta": f"[GPT-4] Respuesta a: {prompt}"}


async def claude_3_call(prompt: str) -> dict:
    # Simula ~1.5 s de latencia de red hacia la API de Anthropic
    await asyncio.sleep(1.5)
    return {"modelo": "claude-3", "respuesta": f"[Claude 3] Respuesta a: {prompt}"}


async def local_llama_call(prompt: str) -> dict:
    # Modelo local: sin salto a internet, solo ~0.5 s de inferencia
    await asyncio.sleep(0.5)
    return {"modelo": "llama-local", "respuesta": f"[Llama local] Respuesta a: {prompt}"}


MODELOS: dict[str, Llamada] = {
    "gpt-4": gpt_4_call,
    "claude-3": claude_3_call,
    "llama-local": local_llama_call,
}


async def con_log(etiqueta: str, llamada: Llamada, prompt: str) -> dict:
    log(f"→ {etiqueta} inicia")
    resultado = await llamada(prompt)
    log(f"← {etiqueta} termina")
    return resultado


async def ejecutar_con_timeout_total(
    llamadas: dict[str, Llamada], prompt: str, timeout: float
) -> dict[str, dict | None]:
    # Devuelve el resultado de cada llamada, o None si el timeout global la canceló
    tareas = {
        nombre: asyncio.create_task(con_log(nombre, llamada, prompt))
        for nombre, llamada in llamadas.items()
    }
    try:
        async with asyncio.timeout(timeout):
            await asyncio.gather(*tareas.values())
    except TimeoutError:
        log(f"✗ Timeout: se superaron {timeout} s, se cancelan las llamadas pendientes")

    # Las tareas que terminaron antes del timeout conservan su resultado
    return {nombre: None if tarea.cancelled() else tarea.result() for nombre, tarea in tareas.items()}


async def con_semaforo(
    semaforo: asyncio.Semaphore, etiqueta: str, llamada: Llamada, prompt: str, timeout: float
) -> dict:
    log(f"· {etiqueta} en cola")
    # Solo MAX_CONCURRENCIA tareas pueden estar dentro del bloque a la vez;
    # el resto espera aquí sin bloquear el event loop
    async with semaforo:
        # El timeout individual corre desde que la llamada consigue lugar: el tiempo
        # en cola no cuenta, y una llamada colgada libera su lugar al vencer
        try:
            async with asyncio.timeout(timeout):
                return await con_log(etiqueta, llamada, prompt)
        except TimeoutError:
            log(f"✗ {etiqueta} superó su timeout de {timeout} s")
            raise


async def ejecutar_con_semaforo(
    llamadas: list[tuple[str, Llamada]], prompt: str, max_concurrencia: int, timeout_por_llamada: float
) -> list[dict | BaseException]:
    semaforo = asyncio.Semaphore(max_concurrencia)
    # return_exceptions=True: el TimeoutError de una llamada se devuelve como
    # resultado en vez de cancelar a las demás
    return await asyncio.gather(
        *(con_semaforo(semaforo, etiqueta, llamada, prompt, timeout_por_llamada) for etiqueta, llamada in llamadas),
        return_exceptions=True,
    )


async def demo_timeout(prompt: str) -> None:
    print(f"=== Resiliencia: timeout de {TIMEOUT_TOTAL} s a la ejecución total ===")
    resultados = await ejecutar_con_timeout_total(MODELOS, prompt, TIMEOUT_TOTAL)

    print()
    for nombre, resultado in resultados.items():
        if resultado is None:
            print(f"{nombre:<12} -> ERROR: cancelada por timeout")
        else:
            print(f"{nombre:<12} -> {resultado['respuesta']}")


async def demo_semaforo(prompt: str, total_llamadas: int = 10) -> None:
    print(
        f"=== Control de flujo: {total_llamadas} llamadas, máximo {MAX_CONCURRENCIA} a la vez, "
        f"timeout de {TIMEOUT_POR_LLAMADA} s por llamada ==="
    )
    modelos = list(MODELOS.items())

    llamadas = []
    for i in range(total_llamadas):
        nombre, llamada = modelos[i % len(modelos)]
        llamadas.append((f"#{i + 1:02d} {nombre}", llamada))

    inicio = time.perf_counter()
    resultados = await ejecutar_con_semaforo(llamadas, prompt, MAX_CONCURRENCIA, TIMEOUT_POR_LLAMADA)
    fallidas = sum(isinstance(r, TimeoutError) for r in resultados)
    log(
        f"✓ {len(resultados) - fallidas} completadas, ✗ {fallidas} por timeout, "
        f"en {time.perf_counter() - inicio:.2f} s"
    )


async def main():
    prompt = "¿Qué es un LLM?"
    await demo_timeout(prompt)
    print()
    # El programa sigue aunque la demo anterior haya terminado en timeout
    await demo_semaforo(prompt)


if __name__ == "__main__":
    asyncio.run(main())
