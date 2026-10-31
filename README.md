# Curso AI Engineering

Ejercicios y notebooks del curso. Requiere **Python 3.12 o superior** (probado con 3.12.16 y 3.14.7).

## Estructura

```
ejercicios/
  01_asyncio/
    simulacion_llamadas.py        # Simulación de llamadas a LLMs: gather, timeout y semáforo
    test_simulacion_llamadas.py   # Pruebas con pytest-asyncio
notebooks/                        # Notebooks .ipynb
```

Cada tema nuevo va en `ejercicios/NN_tema/`, con el número siguiente.

## Instalación

Con [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Con pip (`--group` requiere pip 25.1 o superior):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install .
pip install --group dev    # pytest, pytest-asyncio, ipykernel
```

## Ejercicio 01: orquestación asíncrona de llamadas a LLMs

[`simulacion_llamadas.py`](ejercicios/01_asyncio/simulacion_llamadas.py) simula tres modelos con distinta latencia de red. La espera usa `await asyncio.sleep()`, nunca `time.sleep`, que bloquearía el event loop.

| Modelo | Corrutina | Latencia simulada |
|---|---|---|
| GPT-4 | `gpt_4_call` | 2.5 s |
| Claude 3 | `claude_3_call` | 1.5 s |
| Llama local | `local_llama_call` | 0.5 s |

El script corre dos demos seguidas:

1. **Resiliencia.** Las tres llamadas se disparan a la vez con `asyncio.gather` dentro de `asyncio.timeout(2)`. GPT-4 supera el límite: el `TimeoutError` se captura, se muestra el error y el programa sigue, conservando las respuestas que llegaron a tiempo.
2. **Control de flujo.** Se disparan 10 llamadas, pero un `asyncio.Semaphore(2)` deja ejecutar solo 2 a la vez. Además, cada llamada tiene su propio timeout de 2 s, que empieza a correr cuando consigue lugar en el semáforo (el tiempo en cola no cuenta). Las fallas individuales se recogen con `gather(..., return_exceptions=True)` sin cancelar al resto.

### Ejecutar

```bash
uv run ejercicios/01_asyncio/simulacion_llamadas.py
# o, con el entorno activado:
python ejercicios/01_asyncio/simulacion_llamadas.py
```

### Salida de ejemplo (Python 3.12.16)

```
=== Resiliencia: timeout de 2.0 s a la ejecución total ===
[ 0.00s] → gpt-4 inicia
[ 0.00s] → claude-3 inicia
[ 0.00s] → llama-local inicia
[ 0.50s] ← llama-local termina
[ 1.50s] ← claude-3 termina
[ 2.00s] ✗ Timeout: se superaron 2.0 s, se cancelan las llamadas pendientes

gpt-4        -> ERROR: cancelada por timeout
claude-3     -> [Claude 3] Respuesta a: ¿Qué es un LLM?
llama-local  -> [Llama local] Respuesta a: ¿Qué es un LLM?

=== Control de flujo: 10 llamadas, máximo 2 a la vez, timeout de 2.0 s por llamada ===
[ 2.00s] · #01 gpt-4 en cola
[ 2.00s] → #01 gpt-4 inicia
[ 2.00s] · #02 claude-3 en cola
[ 2.00s] → #02 claude-3 inicia
[ 2.00s] · #03 llama-local en cola
[ 2.00s] · #04 gpt-4 en cola
[ 2.00s] · #05 claude-3 en cola
[ 2.00s] · #06 llama-local en cola
[ 2.00s] · #07 gpt-4 en cola
[ 2.00s] · #08 claude-3 en cola
[ 2.00s] · #09 llama-local en cola
[ 2.00s] · #10 gpt-4 en cola
[ 3.50s] ← #02 claude-3 termina
[ 3.50s] → #03 llama-local inicia
[ 4.00s] ✗ #01 gpt-4 superó su timeout de 2.0 s
[ 4.00s] → #04 gpt-4 inicia
[ 4.00s] ← #03 llama-local termina
[ 4.00s] → #05 claude-3 inicia
[ 5.51s] ← #05 claude-3 termina
[ 5.51s] → #06 llama-local inicia
[ 6.00s] ✗ #04 gpt-4 superó su timeout de 2.0 s
[ 6.00s] → #07 gpt-4 inicia
[ 6.01s] ← #06 llama-local termina
[ 6.01s] → #08 claude-3 inicia
[ 7.51s] ← #08 claude-3 termina
[ 7.51s] → #09 llama-local inicia
[ 8.01s] ✗ #07 gpt-4 superó su timeout de 2.0 s
[ 8.01s] → #10 gpt-4 inicia
[ 8.01s] ← #09 llama-local termina
[10.01s] ✗ #10 gpt-4 superó su timeout de 2.0 s
[10.01s] ✓ 6 completadas, ✗ 4 por timeout, en 8.01 s
```

Qué muestra la salida:

- **Inicio casi simultáneo:** las tres llamadas arrancan en `[ 0.00s]`, y las 10 de la segunda demo entran a la cola en `[ 2.00s]`.
- **Timeout global:** a los 2.00 s se captura `TimeoutError`. GPT-4 se cancela; Claude y Llama conservan su respuesta.
- **Semáforo:** nunca hay más de 2 llamadas en ejecución. Cada vez que una termina (`←`) o falla (`✗`), entra la siguiente de la cola (`→`).
- **Timeout por llamada:** cada GPT-4 falla a los 2 s de *empezar*, no de entrar a la cola, y libera su lugar.

### Pruebas

```bash
uv run pytest
```

| Prueba | Qué verifica |
|---|---|
| `test_las_llamadas_corren_en_paralelo` | Tres llamadas de 0.1, 0.2 y 0.3 s terminan en menos de 0.45 s (en serie serían 0.6 s): no hay bloqueo. |
| `test_timeout_total_cancela_la_lenta_y_conserva_las_demas` | El timeout global cancela la llamada lenta y mantiene el resultado de la rápida. |
| `test_semaforo_limita_la_concurrencia` | Con 10 llamadas y `Semaphore(2)`, el máximo de llamadas activas a la vez es exactamente 2. |
| `test_timeout_por_llamada_no_afecta_a_las_demas` | Una llamada que vence su timeout devuelve `TimeoutError` sin afectar a la otra. |
| `test_el_tiempo_en_cola_no_cuenta_para_el_timeout` | Una llamada que espera en cola no vence por ese tiempo de espera. |

Las pruebas usan llamadas falsas de décimas de segundo, así que corren en un segundo aproximadamente.
