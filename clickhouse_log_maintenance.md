# ClickHouse System Log Maintenance

This guide covers setting up log rotation (TTL) and reclaiming disk space for ClickHouse system logs.

## Prerequisites

- Access to the Docker host running ClickHouse
- Container name: `euf_search_clickhouse`
- Run as root or with sudo privileges

---

## Quick Start

### Step 0: Check Current Disk Usage (Optional)

See which tables are consuming the most space before making changes:

```bash
docker exec euf_search_clickhouse clickhouse-client -q "
SELECT
  database,
  table,
  formatReadableSize(sum(bytes)) AS size,
  sum(rows) AS rows
FROM system.parts
WHERE active
GROUP BY database, table
ORDER BY sum(bytes) DESC
LIMIT 20
"
```

---

### Step 1: Set TTL (Rolling Retention)

Set 30-day automatic retention on heavy system log tables. 

**Note:** No `-it` flag (avoids TTY issues with Ctrl+C):

```bash
# System trace logs
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.trace_log MODIFY TTL event_date + INTERVAL 30 DAY"

# Text logs (errors, warnings)
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.text_log MODIFY TTL event_date + INTERVAL 30 DAY"

# Background schedule pool logs
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.background_schedule_pool_log MODIFY TTL event_date + INTERVAL 30 DAY"

# Part logs (merge/mutation operations)
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.part_log MODIFY TTL event_date + INTERVAL 30 DAY"

# Metric logs (can be very large)
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.metric_log MODIFY TTL event_date + INTERVAL 30 DAY"

# Asynchronous metric logs
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.asynchronous_metric_log MODIFY TTL event_date + INTERVAL 30 DAY"

# Query logs (often the largest - tracks all queries)
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.query_log MODIFY TTL event_date + INTERVAL 30 DAY"

# Query thread logs
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.query_thread_log MODIFY TTL event_date + INTERVAL 30 DAY"
```

---

### Step 2: Verify TTL is Applied

Check that TTL settings are correctly stored:

```bash
docker exec euf_search_clickhouse clickhouse-client -q "
SELECT
  database,
  name AS table,
  engine_full
FROM system.tables
WHERE database='system'
  AND name IN
    ('trace_log','text_log','background_schedule_pool_log','part_log',
     'metric_log','asynchronous_metric_log','query_log','query_thread_log')
ORDER BY table
"
```

You should see `TTL event_date + toIntervalDay(30)` in the `engine_full` column.

---

### Step 3: Handle MEMORY_LIMIT_EXCEEDED Errors (If Needed)

If an `ALTER` fails with `MEMORY_LIMIT_EXCEEDED` (common for `metric_log`):

1. **Truncate first** (removes backlog, preserves table structure):
   ```bash
   docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.metric_log"
   ```

2. **Then apply TTL** (survives truncation):
   ```bash
   docker exec euf_search_clickhouse clickhouse-client -q \
   "ALTER TABLE system.metric_log MODIFY TTL event_date + INTERVAL 30 DAY"
   ```

---

### Step 4: Reclaim Disk Space Immediately

**WARNING:** This permanently deletes all existing log data in these tables. Only run if you're sure you don't need historical logs.

```bash
# Truncate heavy backlog tables (frees 40+ GiB quickly)
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.trace_log"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.text_log"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.background_schedule_pool_log"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.part_log"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.metric_log"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.asynchronous_metric_log"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.query_log"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.query_thread_log"
```

---

### Step 5: Confirm Space Reclaimed

Verify system log tables are now small:

```bash
docker exec euf_search_clickhouse clickhouse-client -q "
SELECT
  table,
  formatReadableSize(sum(bytes)) AS size,
  sum(rows) AS rows
FROM system.parts
WHERE active AND database='system'
GROUP BY table
ORDER BY sum(bytes) DESC
"
```

---

### Step 6: Check Docker Volume Usage

See overall Docker disk usage:

```bash
# All volumes
docker system df -v

# ClickHouse-specific volumes only
docker system df -v | grep -i clickhouse
```

---

## Troubleshooting

### Issue: ALTER fails with MEMORY_LIMIT_EXCEEDED

**Solution:** Truncate first, then apply TTL (see Step 3).

The error occurs because ClickHouse tries to rewrite large tables in memory. Truncating removes the data without needing much memory, then TTL is applied to future data.

### Issue: "No such table" errors

**Solution:** Some tables may not exist in your ClickHouse version. This is normal - skip those tables.

Check available system tables:
```bash
docker exec euf_search_clickhouse clickhouse-client -q "SHOW TABLES FROM system"
```

### Issue: Need to change retention period

Modify the interval in the ALTER command:

```bash
# 7 days instead of 30
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.query_log MODIFY TTL event_date + INTERVAL 7 DAY"

# 90 days
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.query_log MODIFY TTL event_date + INTERVAL 90 DAY"
```

---

## Additional System Tables (Optional)

If disk space is still tight, consider these:

```bash
# Session logs
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.session_log MODIFY TTL event_date + INTERVAL 30 DAY"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.session_log"

# Crash logs
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.crash_log MODIFY TTL event_date + INTERVAL 30 DAY"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.crash_log"

# ZooKeeper logs (if using ZooKeeper)
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.zookeeper_log MODIFY TTL event_date + INTERVAL 30 DAY"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.zookeeper_log"

# Processes logs
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.processes_log MODIFY TTL event_date + INTERVAL 30 DAY"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.processes_log"

# Transactions logs
docker exec euf_search_clickhouse clickhouse-client -q \
"ALTER TABLE system.transactions_info_log MODIFY TTL event_date + INTERVAL 30 DAY"
docker exec euf_search_clickhouse clickhouse-client -q "TRUNCATE TABLE system.transactions_info_log"
```

---

## Automated Cleanup (Cron Job)

To run this monthly, add to crontab on your host:

```bash
# Edit crontab
sudo crontab -e

# Add line (runs 1st of every month at 2 AM)
0 2 1 * * /path/to/clickhouse_log_maintenance.sh >> /var/log/clickhouse_maintenance.log 2>&1
```

Or create a systemd timer for more robust scheduling.

---

## Verify Current Settings

Check TTL on a specific table anytime:

```bash
docker exec euf_search_clickhouse clickhouse-client -q "SHOW CREATE TABLE system.query_log"
```

Look for `TTL event_date + toIntervalDay(30)` in the output.

---

## Notes

- **TTL** = Time To Live; automatically deletes old rows
- **TRUNCATE** = Immediate deletion of all rows (cannot be undone)
- TTL only affects **new data** unless combined with mutations
- These system logs are safe to truncate; they don't contain business data
