"""atc — API traffic control for the T1 API (0.73.0 pre-alpha 1).

Manages, optimises, end-to-end encrypts and protects the traffic that
reaches a T1 server:

* :mod:`.shield` -- notices attackers by behaviour (probing, key guessing,
  API mapping) and jails them with growing timeouts;
* :mod:`.traffic` -- in-flight limits per server and per client, and
  per-route statistics;
* :mod:`.e2e` -- ``hypernix-atc/1`` end-to-end encryption: a handshake
  signed by the server's identity key, sealed requests and framed sealed
  replies;
* :mod:`.config` -- the ``T1_ATC_*`` settings.

Work in progress: the ASGI middleware and the ``/atc`` routes that put
these in front of the server come next; until then nothing here is
loaded by the T1 API.
"""
