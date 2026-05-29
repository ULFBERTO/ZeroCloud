  readonly models = this.modelManager.models;
  readonly isWebLLMSupported = this.webllm.isSupported;

  // Lógica para destacar modelos Oxide
  readonly oxideModels = computed(() =>
    this.models().filter(m => 
      m.name?.toLowerCase().includes('oxide') || m.repoId?.toLowerCase().includes('oxide')
    )
  );

  readonly otherModels = computed(() =>
    this.models().filter(m => 
      !m.name?.toLowerCase().includes('oxide') && !m.repoId?.toLowerCase().includes('oxide')
    )
  );

  // Imports