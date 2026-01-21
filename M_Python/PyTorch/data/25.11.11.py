import logging as log

# 关键：把默认等级调到 INFO
log.basicConfig(level=log.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

a = 3
b = 4
c = a + b
r = c ** 2
log.info(f"a={a}, b={b}, c={c}, r = {r}")
log.info("r = {}".format(c))
