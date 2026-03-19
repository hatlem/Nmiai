 running = true;
 try {
 const success = await runPipeline(round.id);
 if (success) completedRounds.add(round.id);
 } catch (e) {
 log(`Pipeline error: ${e.message}`);
 console.error(e);
 } finally {
 running = false;
 }
 }
 } catch (e) {
 log(`Poll error: ${e.message}`);
 }
 }
 function startPoll() {
 if (pollTimer) clearInterval(pollTimer);
 log("Auto-poller started (30s interval). window.__stopPoll() to stop.");
 checkAndRun(); 
 pollTimer = setInterval(checkAndRun, POLL_INTERVAL);
 }
 function stopPoll() {
 if (pollTimer) {
 clearInterval(pollTimer);
 pollTimer = null;
 log("Auto-poller stopped.");
 }
 }
 window.__stopPoll = stopPoll;
 window.__startPoll = startPoll;
 window.__runNow = async function (roundId) {
 if (running) { log("Already running"); return; }
 running = true;
 try {
 await runPipeline(roundId);
 } catch (e) {
 log(`Manual run error: ${e.message}`);
 console.error(e);
 } finally {
 running = false;
 }
 };
 window.__completedRounds = completedRounds;
 startPoll();
})();
