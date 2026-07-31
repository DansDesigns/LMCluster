# LMCluster

LMCluster runs one large language model across several computers at once.

The problem it solves is a simple one. A model you want to run needs more
memory than any single machine you own has, but between them your machines
have plenty. LMCluster pools that memory: one machine loads the model and
the rest lend it whatever they have spare, so a model that fits on none of
them individually will run across all of them together. Four laptops with
32 GB each become something closer to one machine with 120 GB.

It does this using llama.cpp, which can hold different layers of the same
model on different computers and pass the intermediate results between them
over your network. Everything else here — the discovery, the dashboard, the
planning — exists to make that arrangement something you can set up in a
few minutes rather than something you configure by hand every time.

## What to expect before you build a cluster around it

This pools memory. It does not divide up the work, and it is worth being
clear about that before you spend a weekend on it.

The layers of a model run in order, one after another. While the second
machine is working on layers thirty to sixty, the first has already done its
part and is sitting idle waiting. Adding machines therefore buys you
capacity, not speed. If a model already fits comfortably on one of your
machines, spreading it across four will make it slower, because you have
added network round trips without adding any parallelism.

The right reason to use this, then, is that a model is too big. It is not a
way to make a model that already fits run faster.

The dashboard tries to stop you making that mistake. Ask it to spread a
model that would fit on one machine and it will tell you not to bother, and
why.

## Installing it

Run the installer on the first machine:

    ./install.sh --with-rpc          # Linux or macOS
    install.bat --with-rpc           # Windows

It sets up a Python environment, offers to build llama.cpp, and creates a
cluster key. 
You can also choose to enter your own key.
This key is a shared secret deciding which machines belong to
your cluster, and every other machine needs the same one:

    ./install.sh --token <the key from the first machine>

The second thing it does is fetch llama.cpp. This is downloaded from the
llama.cpp project's own published releases, checked against the size and
checksum GitHub publishes for it, takes a minute or two, and needs no
compiler — the archives contain `ggml-rpc-server`, which is the
one program this whole project depends on, along with `llama-server` and
the libraries they need. The installer picks the right archive for your
machine and hardware, checks that what it downloaded actually contains
those programs, and records where they went.

If you would rather compile it yourself, perhaps to get a build tuned to
your own processor, pass `--build-from-source`. That needs a C++ compiler
and cmake, and takes anywhere from a few minutes to half an hour. On
Windows a compiler means either Visual Studio Build Tools with the
"Desktop development with C++" workload, which is several gigabytes, or
MSYS2, which is smaller:

    # Visual Studio Build Tools
    winget install Microsoft.VisualStudio.2022.BuildTools --override \
      "--quiet --wait --add Microsoft.VisualStudio.Workload.NativeDesktop"

    # or MSYS2, from https://msys2.org, then in the UCRT64 shell:
    pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake \
              mingw-w64-ucrt-x86_64-ninja git

Open a new terminal afterwards. Windows only makes newly installed programs
visible to terminals started after the installation, so a window that was
already open will not find the compiler, and the installer will say so
rather than pretending something is broken.

To see what your machine has without changing anything:

    python install.py --check-tools

Depending on your system, you may need to run:

```
python3 <- notice the 3 
```

A few flags worth knowing:

    ./install.sh --with-rpc                 rpc is required for RAM pooling
    ./install.sh --gpu vulkan               force a particular backend
    ./install.sh --build-from-source        compile rather than download
    ./install.sh --yes                      accept every default, no prompts
    ./install.sh --new-token                generate a new  cluster key

Then start the node:

    ./run.sh               # Linux
    run.bat                # Windows

The dashboard is at http://localhost:8470. Every machine runs the same thing
and serves the same dashboard, so you can drive the cluster from whichever
one you happen to be sitting at.

Last, put your `.gguf` files somewhere and tell the dashboard where that is,
under Settings (or where you specified during install).

## How it hangs together

Every machine runs one copy of LMCluster and they are all equals. There is
no master to nominate and nothing to configure about the topology. Each one
broadcasts a small message on your local network every few seconds saying
who it is, how much memory it has spare, whether it is currently lending
that memory, and what sort of network connection it is on. That is how they
find each other, and it is also how the machine loading a model knows what
it has to work with.

When you load a model, the machine you loaded it from becomes the one
holding the conversation. It starts llama.cpp's server with a list of the
other machines' addresses and llama.cpp distributes the layers across them.
Those other machines are doing nothing clever: they hold weights and do
arithmetic when asked.

Because that server speaks the ordinary OpenAI API, the chat page is a thin
client over it. Nothing in the design knows or cares whether the model it is
talking to is spread across four machines or sitting entirely on one.

## The strip along the top

Every page carries a line of machine names with a coloured dot beside each.
Green means that machine is lending its memory. Amber means it is switched
on and contributing nothing. Red means it has stopped announcing itself —
switched off, asleep, or off the network. A small diamond marks whichever
machine is currently holding a model.

A machine that goes quiet stays on the list rather than disappearing,
because a machine silently vanishing is exactly the thing you want to
notice. Hovering over a name says what its colour means.

## The Pool page

Machines are on the left, the pool and the load controls in the middle,
and this machine on the right. It exists to answer one question: how much
memory does this cluster have, and why isn't it more?

At the top is the pool — the usable memory across the machines currently
contributing, and beside it, in a different colour, how much more you would
have if the rest joined in.

Below that is every machine on the cluster, and this is the part worth
explaining. A machine can sit there looking perfectly healthy and still
contribute nothing, and there are several quite different reasons why, so
the dashboard names the reason rather than showing you a red dot and leaving
you to guess.

If a machine has llama.cpp built but simply isn't lending its memory yet,
there is a button to start it. The same applies if its worker was running
and has since crashed. In both cases LMCluster asks that machine to start
its own worker, using the cluster key to prove it has the right to.

If the connection times out rather than being refused, that is a firewall
quietly discarding the traffic, and no amount of remote poking will fix it.
The dashboard shows you the command to open the port on that machine and
leaves it with you.

And if a machine never had llama.cpp built with RPC support at all, it tells
you to re-run the installer there.

That distinction between a refused connection and one that times out is the
whole reason the fix button can exist. Refused means nothing is listening,
which we can start remotely. A timeout means something is listening but the
packets are going nowhere, which we cannot.

## Network connections, and why they matter here

Intermediate results cross the network on every single token the model
produces, so what each machine is connected by affects everything.

The dashboard works out what that is and says so. A machine on 2.4 GHz Wi-Fi
is flagged in red, because that band is shared with every microwave and
neighbouring router around and in practice manages somewhere between ten and
fifty megabits. It will hold up the whole cluster, and its owner usually has
no idea, because as far as they can tell it has Wi-Fi and is working fine.
The advice is to move it to your 5 GHz network or, better, plug it in.

On 5 GHz or 6 GHz it will say the machine is fine, while noting that
ethernet is steadier, since Wi-Fi latency wanders with interference and
every token pays for that. If it cannot tell which band a machine is on it
says so and suggests you check, because plenty of routers publish both bands
under one name and a machine can quietly end up on the wrong one.

Wired machines get checked too. Ethernet running at 100 Mbps rather than a
gigabit is almost always an old Cat5 cable, a 100 Mbps port on a switch, or
a bad crimp, and swapping the cable usually sorts it.

The pool summary also names the single slowest connection in the cluster,
because when the layers run in order every machine ends up waiting on the
slowest hop.

## When a machine cannot be seen

The commonest way a cluster half-works is a firewall. Three ports have to
be reachable on every machine: 8470 for the dashboard and the requests one
machine makes of another, 8471 for the announcements they use to find each
other, and 50052 for lending memory.

A blocked port produces symptoms that point somewhere else entirely. Block
the announcements and a machine simply never appears, so it looks switched
off. Block the RPC port and it appears perfectly healthy while contributing
nothing.

Settings has a Network ports section which shows the state of all three and
opens them for you. On Windows that raises the usual administrator prompt;
on Linux it asks for your password through the desktop, or uses sudo if the
machine is set up for it. On a headless machine, where there is no prompt to
answer, it gives you the exact commands to paste over SSH. The installer
offers the same thing while you are already sitting there.

Two things are worth knowing here, because both look like firewall
problems and neither is.

The first is that machines will often disagree about the size of the pool,
and this is usually correct rather than a fault.

A machine always counts its own memory, because whichever machine you load
a model from holds part of that model itself — it does not have to lend
anything to anybody to do that. But it counts another machine only if that
machine is lending and can be reached. So if one of three machines has not
joined the pool, that machine will show a larger total than the other two:
it is counting itself, and they are not counting it.

The pool panel writes the arithmetic out, one line per machine, so you can
see which machines went into the total rather than having to take it on
trust. The figure that matters is the one on the machine you actually load
from.

The second If one machine can see the whole cluster while
the others cannot see it, the problem is usually that its announcements are
going out on the wrong network adapter — a machine with WSL, Hyper-V,
VirtualBox or Docker installed has several, and a broadcast sent to the
network in general leaves by whichever one the routing table prefers.
LMCluster now sends its announcements on every adapter, and also answers
directly to any machine it has heard from, which repairs this by itself:
if your broadcasts are not reaching a machine but its broadcasts reach you,
you start answering it personally and the two become visible to each other.

On Windows there is one more trap. Rules are written per network profile,
and Windows decides for itself whether your network is Public or Private —
frequently choosing Public for a perfectly ordinary home network. A rule
written for Private networks then does nothing at all, while looking
entirely correct in the firewall settings. LMCluster writes its rules for
every profile to sidestep this, and the Network ports panel tells you if
Windows has classified the network as Public, which is worth changing
regardless.

## Updates

Settings has an Updates section which compares your version against the one
in the repository. Installing downloads the current code into `tmp/`, checks
it actually is LMCluster before touching anything, keeps a copy of your
present version in `tmp/backup`, and then restarts the node. Your settings,
cluster key and downloaded llama.cpp are left alone.

The page you are looking at waits for the node to come back and reloads
itself, rather than opening a new tab. It waits on the process rather than
the version number, because the new `version.txt` is written a second or two
before the process is actually replaced, and a page watching the version
would reload into a server that was about to exit.

## The cluster key

The key is the shared secret deciding which machines are part of your
cluster. You will find it under Settings, with a button to copy it and
another to replace it.

Replacing it is a little delicate, so it is worth knowing what happens. The
new key goes out to every machine that is switched on first, and only then
does the machine you are sitting at change its own. That order matters: if
it changed its own key first it would immediately lose the authority to tell
anybody else, and the cluster would fall apart.

Machines that are asleep or switched off cannot be told anything, so they
keep the old key and drop out. There is no way around that, so instead of
reporting a clean success the dashboard names them, and you can either give
them the new key from their own dashboards or re-run their installers with
it.

Reading or changing the key only works from a browser on the machine itself,
not from across the network. Protecting those pages with the key would be
circular, since you would need the secret in order to be shown the secret,
and leaving them open would let anybody on your network read your cluster
key out of the dashboard.

One thing to be honest about: the key is a boundary around your local
network rather than real security. It travels in the clear over ordinary
HTTP, so it keeps well-behaved machines apart rather than defending against
somebody already listening to your traffic. Run the cluster on a network you
trust.

That applies with more force to the layer-sharing itself, and this is worth
reading twice. llama.cpp's own documentation describes its RPC backend as
being at a proof-of-concept stage, says the functionality is fragile and
insecure, and tells you never to run the RPC server on an open network or in
a sensitive environment. LMCluster's cluster key does nothing to change
that, because it protects LMCluster's own dashboard and API and not the port
llama.cpp itself listens on. Anything that can reach that port can ask the
machine to do work. Keep this on your home network, behind your router, and
do not forward those ports to the internet.

## Skills

A skill is a small Python file that does one thing, which the cluster can
run for you. It has a docstring describing what it takes and returns, and a
single `run(inputs)` function:

    """
    Skill: Word Count
    Description: Count words, characters and lines in some text
    Version: 1.0
    Author: cluster
    Inputs: {"text": "string"}
    Outputs: {"words": "number", "characters": "number"}
    Tags: text
    """


    def run(inputs):
        text = inputs.get("text", "")
        return {"words": len(text.split()), "characters": len(text)}

You can write these yourself, or describe what you want on the Skills page
and have the loaded model write one. When it does, the result comes back
into the editor for you to read before anything is saved, which is
deliberate, and brings us to the warning.

Skills are not sandboxed. A skill is ordinary Python running with your user
account's permissions. It runs in a separate process, so one stuck in an
infinite loop gets killed when its timeout expires instead of freezing the
node, and one that crashes doesn't take the node down with it — but that is
protection against accidents, not against malice. Being able to save a skill
amounts to having a shell on that machine. The checking done before a skill
is saved catches syntax errors and obvious mistakes; it is not a security
boundary and could not be made into one. Treat skills you did not write
yourself, including ones a model wrote, as code you are choosing to trust.

## Settings

The Settings page covers three separate things, and the distinction matters
because they take effect at different times.

**How the model behaves** applies to your next message, with no reloading.
This is where the system prompt lives — the instruction sent ahead of every
conversation to set the model's manner — along with temperature, top-p,
top-k, min-p, repetition penalty and a reply length limit. A blank system
prompt is a real choice rather than an empty field, and some models do
behave better without one. The reply limit is worth setting on a slow
cluster, where an answer that rambles for a thousand tokens can occupy the
whole thing for ten minutes.

**How models get loaded** applies the next time you load one, and does
nothing to a model already running. The context window is here, and it is
worth understanding that it is not free: the memory it needs is taken on
every machine, out of the same budget the model itself is competing for.

Also here is the share of the model each machine takes. Left blank,
llama.cpp divides the layers according to how much memory each machine has
free, which is usually the right answer. You would set it by hand when
memory is a poor guide to how useful a machine is — an old laptop with
plenty of free RAM and a slow processor will happily accept a third of the
model and then hold everything else up, and giving it `1` where a faster
machine gets `3` fixes that. The numbers are one per machine, this one
first, and they are checked against the machines actually in the pool
before anything is loaded rather than after llama.cpp has buried a complaint
in several hundred lines of startup output.

**This machine** holds its name on the cluster, the model folder, and
whether it lends its memory automatically on startup.

## Configuration

The installer writes `lmcluster.toml` and you can mostly leave it alone. The
parts you might want to change:

    [node]
    name = ""              # blank means use the hostname
    port = 8470

    [cluster]
    require_token = true   # refuse requests from machines without the key

    [discovery]
    port = 8471            # must be the same on every machine
    interval = 3.0         # seconds between announcements
    timeout = 15.0         # a machine counts as gone after this long

    [chat]
    system_prompt = ""     # sent ahead of every conversation
    temperature = 0.7
    top_p = 0.95
    top_k = 40
    min_p = 0.05
    repeat_penalty = 1.1
    max_tokens = 0         # 0 means no limit beyond the context window

    [shard]
    rpc_port = 50052              # where this machine listens when lending
    ctx = 4096                    # context window to load models with
    tensor_split = ""             # share of the model per machine
    extra_args = ""               # anything else to pass to llama.cpp
    master_port = 8080
    auto_start_worker = true      # lend memory as soon as this machine starts
    rpc_server = "/path/to/rpc-server"
    llama_server = "/path/to/llama-server"
    model_dir = "/where/your/gguf/files/are"

The two binary paths get filled in by the installer once it has built
llama.cpp. If you re-run the installer it updates them in place and leaves
everything else, your comments included, exactly as it found them.

## The HTTP API

Everything the dashboard does is available over HTTP, so you can drive a
cluster from a script.

Some of these want the cluster key in an `X-Cluster-Token` header. Anything
that changes how a node is configured will accept either that key or a
request coming from the machine itself, which is how the dashboard manages
without knowing the secret. The three endpoints dealing with the key only
work from the machine itself.

Reading the state of things:

    GET  /api/health          this machine: name, hardware, whether it can shard
    GET  /api/pool            the cluster: memory, every machine, what is wrong
    GET  /api/models          .gguf files in your model folder
    GET  /api/model           what is loaded, and across how many machines
    GET  /api/load/log        what llama.cpp has printed since it started
    GET  /api/chats           saved conversations
    GET  /api/skills          the skills on this machine

Loading and unloading:

    POST /api/plan            work out where a model would go, without loading
    POST /api/load            load a model across the pool
    POST /api/unload          stop it
    POST /api/pool/offer/start          lend this machine's memory
    POST /api/pool/peers/{id}/start     start another machine's worker

Using it:

    POST /api/chat            streamed reply, newline-delimited JSON
    POST /api/skills/generate have the model write a skill
    POST /api/skills/{id}/run run one

Planning before you load is worth doing, because the explanation that comes
back is written to be read:

    curl -s localhost:8470/api/plan \
         -H 'Content-Type: application/json' \
         -d '{"model_path":"/models/qwen3-235b-q4.gguf"}' | jq -r .explanation

which will say something along the lines of:

    60.0 GB model across this node (17.9 GB) plus 2 worker(s): desk, mac.
    Pooled: 79.6 GB. Expect capacity, not speed: layers run in sequence.

or, if you are asking too much of it:

    400.0 GB model across this node plus 3 worker(s). Pooled: 103.4 GB.
    Still short by 296.6 GB — it will spill to disk and crawl, or fail to
    load.

## What the files are

    install.py            the installer, including the llama.cpp build
    lmcluster/
      config.py           reads lmcluster.toml, keeps this node's identity
      auth.py             the cluster key, and who may do what
      discovery.py        the announcements machines send each other
      hardware.py         memory, GPU and network connection detection
      rpc.py              deciding where layers go, and running llama.cpp
      engine.py           talking to the loaded model
      skills.py           storing and running skills
      store.py            saved conversations
      node.py             the node itself, and its API
      static/index.html   the dashboard

## Known limitations

Some of these I would fix given more time. Others are inherent.

The planner is greedy. It sorts machines by how much memory they have spare
and takes them until the model fits. It does not measure how fast the
connection to each one actually is, so on a mixed cluster it will happily
enlist a machine on Wi-Fi ahead of a wired one that had slightly less memory
free. Measuring the links and weighing them accordingly would be the first
real improvement.

Layer placement is left entirely to llama.cpp. LMCluster tells it which
machines are available and lets it distribute; it does not assign particular
layers to particular machines, which would give better results on a cluster
where the machines differ a lot from one another.

A machine that is switched off cannot be given a new cluster key. That is
inherent rather than an oversight, and the dashboard tells you which
machines it missed so you can deal with them yourself.

Network detection has only really been exercised on Linux. The Windows and
macOS versions are written against the documented output of `netsh` and
`system_profiler` and will want testing on a real machine of each.

Nothing here has yet run against a genuinely large model on real separate
machines. Two nodes have been run together properly, including finding each
other, starting one machine's worker from another, loading a model,
streaming a reply, and having the model write a skill that then ran
correctly — but with a stand-in for llama.cpp rather than llama.cpp itself.
The command LMCluster builds for it is tested for correctness and has not
been executed in anger.
