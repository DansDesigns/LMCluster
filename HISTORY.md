# How this project got here

This is a record of the decisions behind LMCluster, kept mainly so that
whoever reads the code in six months can argue with the reasoning rather
than having to reconstruct it.

## It started as two projects

There were two separate things, LMCluster and Round-Table, both trying to
do something with several machines and local language models, and they were
merged.

Round-Table became the foundation. That was not a close call. It was built
on FastAPI with proper async, streamed over WebSockets, had a sensible
configuration system with a persistent identity for each node, and kept its
data in SQLite with write-ahead logging. The original LMCluster was Flask
with threads, called Ollama directly from inside its route handlers, and
wrote the list of currently-visible machines to a database table every few
seconds — which is odd, because that information is worthless fifteen
seconds later and had no business being on disk.

The name LMCluster survived because it describes what the thing does. The
"round table" idea, which was several models discussing a question with each
other, has since been removed entirely, for reasons below.

Three things came across from the original LMCluster and are still here.

The first was authentication, and it mattered most. Round-Table had none at
all. Any process on the local network could answer a discovery broadcast and
be handed work, or post to its inference endpoint and spend somebody else's
hardware. The shared-key idea from LMCluster fixed that and is now in
`auth.py`.

The second was skills, which Round-Table had no equivalent of.

The third was that the installer knew how to check for and install things,
rather than just making a virtual environment and hoping.

## Then most of it was removed

The merged project could do two quite different things. It could run a
separate model on every machine and combine their answers, and it could run
one model spread across all of them. Those sound related and are not.

The second one was the point. The first was a different project wearing the
same clothes: it needed a model small enough to fit on each machine
individually, which is the exact opposite constraint, and it dragged in a
whole abstraction layer of interchangeable inference backends to support
running different programs on different machines.

So it went. Council mode, the workshop mode built on top of it, the ability
for external agents to register themselves as participants, and the entire
backends package all came out. What replaced the backends package is
`engine.py`, one file that talks to the one llama.cpp server this node
started. It is shorter than the abstract base class it replaced.

Ollama went with it. The reasoning is worth stating plainly, because it
initially looked like a useful thing to keep: Ollama cannot split a model
across machines. It has no equivalent of llama.cpp's RPC backend. So for
this project it was never able to do the main job, and the only job it could
do — juggling several models on one machine — was a job that only existed
because of the mode that has now been removed.

There is also a second reason it would have gone anyway. llama.cpp added a
model router at the end of 2025, which loads and unloads models on demand
behind one endpoint. That was the last thing Ollama was doing here that
llama.cpp could not.

## Bugs worth remembering

Three of these were found by testing rather than by reading, which is the
argument for testing.

**Skill execution had a fake sandbox.** The original swapped in a
restricted set of built-in functions before running a skill, which looks
like a sandbox and is not one: it still exposed `os` and `sys`, and
replacing `__builtins__` does not stop a module importing whatever it
likes. Keeping it would have created false confidence, so it was removed and
the documentation now says plainly that skills are not sandboxed.

**Skill execution also ignored its own timeout.** It accepted a `timeout`
argument and never used it, so a skill containing `while True:` would wedge
the node permanently. The obvious fix was to run the skill on a worker
thread and give up on it after the timeout — and that turned out to be
worse. Python's `redirect_stdout`, which is how the skill's output was being
captured, replaces `sys.stdout` for the whole process rather than for one
thread. Capturing a skill's output on a worker thread therefore swallowed
every other thread's output too, including the web server's, for as long as
the skill ran, which on a skill stuck in a loop meant forever. The
symptom was a test that printed nothing at all.

Running skills in a subprocess fixes the timeout, the output capture, and
crash isolation, all at once. That is what it does now.

**Settings could be changed by anybody.** The endpoint that updates a
node's configuration accepts a new cluster key among its fields, and it had
no guard on it, so any machine on the local network could rewrite a node's
key and walk it into a different cluster without authenticating. This
surfaced because a test of key rotation reported success pushing a new key
to a machine that should have rejected it.

The fix is a second kind of guard. Endpoints that change a node's
configuration now accept either a request from the machine itself, which is
what the dashboard is, or a valid cluster key, which is how one machine
proves it belongs when talking to another. Both callers are legitimate;
neither is the open network. What keeps the loopback half of that safe is
that there is no CORS middleware, so a web page in your browser cannot post
JSON to a node on your behalf.

## Two things the installer got wrong

**It refused to touch an existing configuration file.** If it found one, it
printed a block of TOML for you to paste in by hand. That is irritating in
general and actively bad for the two binary paths it needs to record after
building llama.cpp, because getting those wrong by hand is easy and the
error you get later is opaque. It now merges: missing sections are appended,
the binary paths are rewritten in place after a build, and everything else
including your comments is left exactly as it was, with the previous version
kept alongside as a `.bak` file.

**A missing cmake was a dead end.** It printed instructions and gave up,
which is a strange thing to do when there is a package manager sitting right
there that could install it. It now offers to, through winget, Homebrew, apt
or dnf, and re-checks afterwards. On Windows it also warns that
freshly-installed tools will not be on the current shell's PATH and the
window needs reopening, because otherwise the next attempt fails for a
reason that looks like the first one.

## Two installer bugs found by somebody actually running it

Both of these were reported from real machines after the installer claimed
to have worked, which is the worst kind of bug because it moves the failure
somewhere far away from its cause.

**The RPC server has been renamed.** llama.cpp used to build a program
called `rpc-server` and now builds `ggml-rpc-server`. The installer asked
cmake to build a target that no longer exists and then looked on disk for a
file that was never going to be there, so on Linux it reported that the
build had produced nothing. Both names are now accepted, as a cmake target
and as a filename, and if neither turns up in the usual places the installer
walks the build tree before giving up.

The same rename affected how the worker is started. Its command-line flags
have also shifted between versions, so rather than assuming a set of them,
the worker now runs the binary with `--help` and picks the flags that
version actually accepts. That matters more than it sounds: the RPC server
binds to localhost unless told otherwise, so getting the host flag wrong
produces a worker that starts perfectly happily and is invisible to every
other machine on the network.

**There was no compiler check on Windows.** The check for a C++ compiler
was skipped entirely on Windows, on the assumption that anyone on Windows
would have Visual Studio. A machine without it therefore passed the tool
check, spent a minute cloning llama.cpp, and then failed inside cmake with a
complaint that `nmake` could not be found — which is a genuinely misleading
error, because nmake ships with Visual Studio and cmake only falls back to
it when it can find no other compiler at all. It now looks for Visual Studio
through `vswhere`, then for `cl.exe`, then for MinGW gcc or clang, tells you
which it found, and passes the right generator for it. When it finds none it
says so before downloading anything.

The first attempt at fixing that introduced a third bug of the same family.
Offering to install Visual Studio Build Tools through winget worked, winget
reported success, and the installer then declared the compiler still
missing and printed the entire list of ways to install a compiler — one
line below having installed one. Two things were wrong. The Visual Studio
bootstrapper is a four-megabyte program that kicks off a multi-gigabyte
download and then exits, so under `--quiet` it returns almost immediately
and winget reports success while the compiler is still downloading; the
override now passes `--wait` as well, which makes the bootstrapper block
until the real install has finished. And detection leaned entirely on
`vswhere`, which only reports an installation once the C++ workload is
fully in place. There is now a plain filesystem check for the toolchain
directory alongside it, so an install that vswhere has not caught up with
is still found.

There is also a distinction now between a failure and needing a new
terminal. Windows only makes newly installed programs visible to terminals
opened afterwards, so a window that was already open cannot see a compiler
installed into it. That is not a fault, and telling somebody to go and
install a compiler they have just installed is not helpful, so it now says
what has actually happened and what to do about it.

While fixing that I also made the installer stop claiming success it had not
earned. It used to print "Installed." at the end regardless of what had
happened, several lines below whatever had gone wrong, and the configuration
step would report the file "already complete" when it was merely unchanged.
Both now say what is actually true.

## Downloading instead of compiling

All of the trouble above — the missing compiler, the misleading nmake
error, the Visual Studio bootstrapper that returns before it has finished,
the business about new terminals and PATH — existed to serve a step that
turned out to be unnecessary.

The llama.cpp project publishes prebuilt binaries for every release, and
those archives contain `ggml-rpc-server`. I had assumed they might not,
since it is a comparatively obscure tool, and I was not willing to write a
downloader against a guess. Dan settled it by sending me one of the
archives, which contains both programs this project needs.

So downloading is now the default and compiling is the option. The
installer works out which archive suits the machine, fetches it, and then
checks that the programs it wanted are actually inside before recording
anything — because an installer that reports success it has not earned is
the specific failure this project has produced three times already.

One detail worth knowing if you ever touch this code. The executables in
those archives are stubs: `llama-server.exe` is about nine kilobytes and
loads `llama-server-impl.dll`, which is about ten megabytes, from beside
it. Copying the two programs out of the archive on their own would give you
two files that cannot start. The whole folder is therefore kept together
and the recorded paths point into it.

CUDA builds additionally need NVIDIA's runtime libraries, which are
published as a separate archive, so those are fetched too when a CUDA build
is chosen.

## Borrowed from NIGHTRUN

NIGHTRUN is a UEFI-resident LLM runtime that boots a machine straight into
a model with no operating system underneath. It is written in Rust and
shares no code with this project, but its installer does something this one
was not doing: it verifies its downloads by SHA-256 against a pinned
revision, and after flashing it reads the media back and compares digests,
so that "verified" means the target actually contains what was intended.

Set against that, the downloader here was fetching a zip over the network
and extracting it with no integrity check whatsoever. It now checks the
size GitHub declares for the asset, checks the SHA-256 digest when GitHub
publishes one, and tests the archive's own internal checksums before
writing anything to disk. Hashing happens while the bytes stream past
rather than in a pass afterwards, which costs nothing.

A missing digest is not treated as a failure, because an older API
response is not evidence of tampering, but the hash is printed either way
so it can be compared between machines.

## The download that ate itself

The first working version of the downloader did this:

    if os.path.isdir(dest):
        shutil.rmtree(dest, ignore_errors=True)
    extract(data, asset["name"], dest)

Two mistakes in three lines, and they compounded.

`ignore_errors=True` meant that a file which could not be deleted was
skipped in silence. The extraction that followed then tried to overwrite
that same file and failed with a permission error — so the problem surfaced
several steps away from its cause, wearing a name that pointed at the wrong
thing entirely. What was actually happening was that a running RPC server
had the library open, and on Windows a file that a process has open cannot
be replaced.

The second mistake was unpacking straight over a working installation. Any
failure part way through leaves you with neither the old version nor the
new one, which is a bad thing to do to somebody who had a working cluster
five seconds earlier.

It now unpacks into a staging folder beside the destination, confirms both
programs are present, and only then swaps it into place. If the swap
cannot happen because something has the old files open, the existing
installation is left exactly as it was and the new version is kept in the
staging folder, so it can be moved by hand rather than downloaded again.

The check for running processes happens before the download rather than
after it, because finding out the folder is locked is not worth thirty
megabytes of waiting. And a locked folder no longer offers to build from
source as a consolation: the download worked perfectly, and it will work
again once whatever is holding the files has been stopped, so suggesting
half an hour of compiling would be an odd response.

## Accusing innocent machines of having firewall problems

Three machines, all seeing each other, all reporting each other as
firewalled. The report was wrong and the reasoning behind it was worse.

The check treated a refused connection as "this machine has not started its
worker" and a timed-out connection as "a firewall is discarding packets".
That reasoning holds on Linux. It does not hold on Windows, where a port
with nothing listening is silently discarded rather than refused — so every
machine that had simply not joined the pool looked, from every other
machine, exactly like a machine behind a firewall.

The fix is to stop guessing when we have been told. Each machine's
announcement says whether it is lending its memory, and it is the authority
on that. If it says it is not, there is nothing to probe and nothing to
diagnose: it has not been asked to join.

When a machine does say it is lending and cannot be reached, the check now
distinguishes a blocked port from an unreachable machine by also trying its
dashboard port. If that answers and the RPC port does not, the block really
is on that one port. If neither answers, something broader is wrong, and
saying "firewall" would send somebody hunting in the wrong place.

The timeout was also far too aggressive. Two seconds with no retry means a
machine busy loading a model, or reached over Wi-Fi, gets reported as
firewalled for being slow. It now retries with a longer timeout.

That made the check slow enough to matter — around fourteen seconds when a
machine is unreachable, since that time is spent waiting for connections to
fail — so it moved into a background loop. The dashboard reads the last
result and answers in about thirty milliseconds instead of blocking, and
anything that changes the answer, such as starting another machine's
worker, forces a fresh look rather than leaving a stale one on screen.

## Pool totals that disagreed

Each machine was reporting a different amount of pooled memory, which was
alarming but correct: the pool is what that machine can reach, and they
could not all reach each other. It was a symptom of the diagnosis bug above
rather than a separate fault, and with that fixed they agree.

It is still a per-machine figure, and now says so. The number that matters
is the one on the machine you load the model from, because that is the
machine which has to reach the others.

## A relative path that only worked one way

A machine reported to the whole cluster that it had no RPC support, having
been installed with `--with-rpc` and having llama.cpp sitting right there.
The same procedure on another machine worked.

The entry point defaulted its config to the bare relative name
`lmcluster.toml`. Started through `run.sh`, which changes directory first,
that resolves correctly. Started any other way — a desktop autostart entry,
a systemd unit, or just running the command from your home directory — it
finds nothing, falls back to built-in defaults, and comes up with sharding
switched off. Nothing looks wrong on the machine itself. It simply tells
everyone else it cannot help.

The default now resolves against the project directory, preferring the
current one only if a config is actually there. The node also prints which
config file it read at startup, since that one line would have answered the
question immediately.

While fixing it I gave `can_shard` a reason rather than a bare no. There
are three quite different ways to end up unable to lend memory — the
setting is off, no path was recorded, or the recorded path points at a
binary that is no longer there — and all three looked identical from
another machine. The third was worse than useless: a stale path reported
the machine as capable and then failed only when somebody tried to use it.

## A note on testing multiple nodes on one machine

Each running copy takes its identity from the port it is on, so two copies
on one machine get distinct identities. The cluster key does not work that
way — it lives in one file per user account — so two copies on one machine
share a key file and will appear to agree with each other no matter what you
do to one of them. Set `XDG_CONFIG_HOME` differently for each when testing.
This has no bearing on a real cluster, where each machine has its own
account and its own file, but it did confound one test before I worked out
what was happening.
