use super::*;

#[test]
fn a_malformed_later_record_prevents_exposing_earlier_commands() {
    let mut bytes = vec![0; COMMAND_BUFFER_HEADER_BYTES + 2 * COMMAND_BYTES];
    CommandBuffer::initialize(&mut bytes, 2, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
    let first = CommandRecord {
        action: Action::DispatchThreadgroups {
            grid: [1; 3],
            local_size: [1; 3],
        },
        ..Default::default()
    };
    first
        .encode_into(
            &mut bytes[COMMAND_BUFFER_HEADER_BYTES..COMMAND_BUFFER_HEADER_BYTES + COMMAND_BYTES],
        )
        .unwrap();
    let second = (COMMAND_BUFFER_HEADER_BYTES + COMMAND_BYTES) / 4;
    for (word, value, error) in [
        (0, 99, RecordError::Action { value: 99 }),
        (1, 2, RecordError::Barrier { value: 2 }),
    ] {
        put_word(&mut bytes, second + word, value);
        assert!(
            matches!(CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE), Err(CommandBufferError::Record { index: 1, error: actual }) if actual == error)
        );
        put_word(&mut bytes, second + word, 0);
    }
    let view = CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
    assert_eq!(
        view.records(0..2).unwrap().collect::<Vec<_>>(),
        [first, CommandRecord::default()]
    );
}

#[test]
fn command_execution_ranges_select_exact_records_and_reject_invalid_bounds() {
    let mut bytes = vec![0; COMMAND_BUFFER_HEADER_BYTES + 3 * COMMAND_BYTES];
    CommandBuffer::initialize(&mut bytes, 3, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
    for index in 0..3 {
        let record = CommandRecord {
            action: Action::DispatchThreads {
                grid: [index as u32 + 1, 1, 1],
                local_size: [1; 3],
            },
            ..CommandRecord::default()
        };
        let start = COMMAND_BUFFER_HEADER_BYTES + index * COMMAND_BYTES;
        record
            .encode_into(&mut bytes[start..start + COMMAND_BYTES])
            .unwrap();
    }
    let view = CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
    for start in 0..=3 {
        for end in start..=3 {
            let records = view.records(start..end).unwrap();
            assert_eq!(records.len(), (end - start) as usize);
            for (index, record) in (start..end).zip(records) {
                assert_eq!(
                    record.action,
                    Action::DispatchThreads {
                        grid: [index + 1, 1, 1],
                        local_size: [1; 3],
                    }
                );
            }
        }
    }
    for (start, end) in [(2, 1), (0, 4), (4, 4), (0, u32::MAX)] {
        assert!(matches!(
            view.records(start..end),
            Err(CommandBufferError::Range { capacity: 3, .. })
        ));
    }
}

#[test]
fn detached_policy_headers_validate_without_allocating_command_records() {
    for capacity in [0, 1, u32::MAX] {
        let mut header = TEST_DESCRIPTOR
            .encode_header(capacity, OWNED_STATE)
            .unwrap();
        assert_eq!(header.len(), COMMAND_BUFFER_HEADER_BYTES);
        TEST_DESCRIPTOR
            .validate_header(&header, capacity, OWNED_STATE)
            .unwrap();
        assert!(matches!(
            TEST_DESCRIPTOR.validate_header(&header, capacity ^ 1, OWNED_STATE),
            Err(CommandBufferError::HeaderPolicyMismatch)
        ));
        let inherited = Inheritance {
            pipeline: StateSource::Parent,
            ..OWNED_STATE
        };
        assert!(matches!(
            TEST_DESCRIPTOR.validate_header(&header, capacity, inherited),
            Err(CommandBufferError::HeaderPolicyMismatch)
        ));
        put_word(&mut header, 1, 7);
        assert_eq!(
            TEST_DESCRIPTOR.validate_header(&header, capacity, OWNED_STATE),
            Err(CommandBufferError::EncodingFault(7))
        );
        assert_eq!(
            TEST_DESCRIPTOR.validate_header(&header[..15], capacity, OWNED_STATE),
            Err(CommandBufferError::Length)
        );
    }
}

#[test]
fn command_buffer_initialization_and_fault_checks_bound_the_record_view() {
    for capacity in [0, 1, 3] {
        let mut bytes = vec![0xa5; COMMAND_BUFFER_HEADER_BYTES + capacity as usize * COMMAND_BYTES];
        CommandBuffer::initialize(&mut bytes, capacity, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
        let view = CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
        assert_eq!(view.records(0..capacity).unwrap().len(), capacity as usize);
        assert!(view
            .records(0..capacity)
            .unwrap()
            .all(|record| record == CommandRecord::default()));
        put_word(&mut bytes, 1, 1);
        assert!(matches!(
            CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE),
            Err(CommandBufferError::EncodingFault(1))
        ));
        CommandBuffer::initialize(&mut bytes, capacity, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
        put_word(&mut bytes, 2, 1);
        assert!(matches!(
            CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE),
            Err(CommandBufferError::HeaderPolicyMismatch)
        ));
    }
    let mut truncated = vec![0xa5; COMMAND_BUFFER_HEADER_BYTES + COMMAND_BYTES - 1];
    assert_eq!(
        CommandBuffer::initialize(&mut truncated, 1, &TEST_DESCRIPTOR, OWNED_STATE),
        Err(CommandBufferError::Length)
    );
    assert!(truncated.iter().all(|&byte| byte == 0xa5));
    assert!(matches!(
        CommandBuffer::decode(&[], &TEST_DESCRIPTOR, OWNED_STATE),
        Err(CommandBufferError::Length)
    ));
    let mut header = [0; COMMAND_BUFFER_HEADER_BYTES];
    put_word(&mut header, 0, u32::MAX);
    assert!(matches!(
        CommandBuffer::decode(&header, &TEST_DESCRIPTOR, OWNED_STATE),
        Err(CommandBufferError::Length)
    ));
}
