export class TrainingSessionConfig {
    id?: number;
    model_id?: number;
    created_at?: number;
    finished_at?: number;
    iterations?: number;
    correct_ratio?: number;
    window_size?: number;
    use_slide?: boolean;
    adjust_pte?: boolean;
    // learning_rate: 0.05;
}