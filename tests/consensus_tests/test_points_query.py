import pathlib

import pytest
import requests

from .assertions import assert_http_ok
from .utils import (
    assert_project_root,
    make_peer_folders,
    start_first_peer,
    start_peer,
    wait_collection_exists_and_active_on_all_peers,
    wait_collection_points_count,
    wait_for_uniform_cluster_status,
    wait_peer_added,
)

N_PEERS = 3
N_SHARDS = 3
N_REPLICA = 2


@pytest.mark.parametrize("consistency", ["all", "majority", "quorum", "none"])
def test_deduplication_and_sorting_with_consistency(
    tmp_path: pathlib.Path, consistency
):
    assert_project_root()
    peer_dirs = make_peer_folders(tmp_path, N_PEERS)
    (bootstrap_api_uri, bootstrap_uri) = start_first_peer(
        peer_dirs[0], "dedup_peer_0_0.log"
    )
    peer_api_uris = [bootstrap_api_uri]
    leader = wait_peer_added(bootstrap_api_uri)
    for i in range(1, N_PEERS):
        peer_api_uris.append(
            start_peer(peer_dirs[i], f"dedup_peer_0_{i}.log", bootstrap_uri)
        )
    wait_for_uniform_cluster_status(peer_api_uris, leader)

    # Create collection
    r = requests.put(
        f"{peer_api_uris[0]}/collections/test_collection",
        json={
            "vectors": {"size": 4, "distance": "Dot"},
            "shard_number": N_SHARDS,
            "replication_factor": N_REPLICA,
        },
    )
    assert_http_ok(r)
    wait_collection_exists_and_active_on_all_peers("test_collection", peer_api_uris)

    # Insert the same point ID into different shards with different vectors/scores
    point_id = 42
    vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.6],
    ]
    for i, vec in enumerate(vectors):
        r = requests.put(
            f"{peer_api_uris[i]}/collections/test_collection/points?wait=true",
            json={"points": [{"id": point_id, "vector": vec}]},
        )
        assert_http_ok(r)

    # Insert some other points for noise
    for i in range(10):
        r = requests.put(
            f"{peer_api_uris[0]}/collections/test_collection/points?wait=true",
            json={
                "points": [{"id": 100 + i, "vector": [i / 10, i / 10, i / 10, i / 10]}]
            },
        )
        assert_http_ok(r)

    # Wait for all points to be replicated
    wait_collection_points_count(peer_api_uris[0], "test_collection", 11)

    # Search for the duplicate point
    search_body = {
        "vector": [0.9, 0.8, 0.7, 0.6],
        "limit": 5,
        "with_payload": False,
        "with_vector": False,
    }
    for uri in peer_api_uris:
        r = requests.post(
            f"{uri}/collections/test_collection/points/search",
            params={"consistency": consistency},
            json=search_body,
        )
        assert_http_ok(r)
        results = r.json()["result"]
        ids = [point["id"] for point in results]
        # Assert no duplicates
        assert len(ids) == len(set(ids)), (
            f"Duplicate IDs found in results with consistency={consistency}"
        )
        # Assert sorted by score descending
        scores = [point["score"] for point in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted by score with consistency={consistency}"
        )
