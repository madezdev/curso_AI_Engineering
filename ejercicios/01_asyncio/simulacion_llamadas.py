import asyncio
import time
from collections.abc import Awaitable, Callable

TIMEOUT_TOTAL = 2.0
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


async def con_semaforo(semaforo: asyncio.Semaphore, etiqueta: str, llamada: Llamada, prompt: str) -> dict:
    log(f"· {etiqueta} en cola")
    # Solo MAX_CONCURRENCIA tareas pueden estar dentro del bloque a la vez;
    # el resto espera aquí sin bloquear el event loop
    async with semaforo:
        return await con_log(etiqueta, llamada, prompt)


async def demo_timeout(prompt: str) -> None:
    print(f"=== Resiliencia: timeout de {TIMEOUT_TOTAL} s a la ejecución total ===")
    tareas = [
        asyncio.create_task(con_log(nombre, llamada, prompt), name=nombre)
        for nombre, llamada in MODELOS.items()
    ]
    try:
        async with asyncio.timeout(TIMEOUT_TOTAL):
            await asyncio.gather(*tareas)
    except TimeoutError:
        log(f"✗ Timeout: se superaron {TIMEOUT_TOTAL} s, se cancelan las llamadas pendientes")

    # Las tareas que terminaron antes del timeout conservan su resultado
    print()
    for tarea in tareas:
        if tarea.cancelled():
            print(f"{tarea.get_name():<12} -> ERROR: cancelada por timeout")
        else:
            print(f"{tarea.get_name():<12} -> {tarea.result()['respuesta']}")


async def demo_semaforo(prompt: str, total_llamadas: int = 10) -> None:
    print(f"=== Control de flujo: {total_llamadas} llamadas, máximo {MAX_CONCURRENCIA} a la vez ===")
    semaforo = asyncio.Semaphore(MAX_CONCURRENCIA)
    modelos = list(MODELOS.items())

    llamadas = []
    for i in range(total_llamadas):
        nombre, llamada = modelos[i % len(modelos)]
        llamadas.append(con_semaforo(semaforo, f"#{i + 1:02d} {nombre}", llamada, prompt))

    inicio = time.perf_counter()
    resultados = await asyncio.gather(*llamadas)
    log(f"✓ {len(resultados)} llamadas completadas en {time.perf_counter() - inicio:.2f} s")


async def main():
    prompt = "¿Qué es un LLM?"
    await demo_timeout(prompt)
    print()
    # El programa sigue aunque la demo anterior haya terminado en timeout
    await demo_semaforo(prompt)


if __name__ == "__main__":
    asyncio.run(main())
